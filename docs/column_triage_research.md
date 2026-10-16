# W4.1 — Column-level triage: literature research

Route W4.1 of `docs/research_program.md` §9. **Research only; no implementation.**

Companion to `docs/literature.md` (R1), which this document does **not** repeat.
R1 established the cell-level landscape (REIN, SAGED, HoloDetect, Raha/Baran,
Picket, CDC) and our novelty position. This document attacks one question R1
did not ask: **given a table, which *column* is worth examining at all?**

Every claim carries a citation (title, venue, year, URL). Anything I could not
verify in a primary source is marked `not found` or **[UNVERIFIED]**. No number
here is estimated unless explicitly labelled *derived*.

Compiled 2026-10-16. **Status: COMPLETE.** Research only — nothing in this
document has been implemented or measured by us. Every number is either cited to
its source or explicitly labelled *derived*.

---

## 0. Executive summary

1. **The problem is named.** "Error localisation" (Fellegi–Holt, JASA 1976, still
   deployed by statistical agencies), "cellwise outlier detection" (robust stats),
   "error attribution/diagnosis" (Data X-Ray, SIGMOD 2015). Use **column-level
   error localisation**. §1
2. **It is also already attacked, by a system R1 under-described.** **ED2
   (CIKM 2019) has a *Column Selector*** that picks which column to label next,
   and its four criteria include per-column cross-validated F1 — the brief's
   "predictability shift". They report it **unstable** and prefer average
   classifier certainty. **Do not claim novelty here.** §2.1b
3. **Our scoring statistic is formally the wrong one.** `1 − P(x | row)` is a
   *density*; Ren et al. (NeurIPS 2019) show density is confounded by background
   frequency and that the correct statistic is a **likelihood ratio** against a
   corruption model. That *is* our `native_country` failure, named and fixed
   elsewhere. §3.1
4. **Rarity ≠ corruption is a definitional gap in the field.** Ruff et al.
   (Proc. IEEE 2021) define *strict anomaly* (other distribution) and *outlier*
   (rare draw from the same one) separately, then say "we make no distinction
   between these terms". We can stand in that gap. §3.2
5. **No published system separates rare-but-valid from corrupted using only the
   single table and no labels.** All four deployed mechanisms import outside
   information: a corpus, labels, an impact model, or a cost model. That bounds
   what W4.5 can achieve. §3.3
6. **Best signal available: per-column contamination fraction.** MPE/PU learning
   (Garg et al., NeurIPS 2021) and outlier-score mixture fitting (Gao & Tan,
   ICDM 2006) independently give the same unsupervised, rarity-immune quantity.
   §2.3, §2.5
7. **G3 is cheap to close.** `pip install raha` and a 6-line script run a
   SIGMOD-reproducibility-award system on our exact `{dirty.csv, clean.csv}`
   layout with no adaptation (repo active 2025-06-05). §4
8. **Build M1 (likelihood ratio) and M2 (contamination fraction) first** — no
   labels, no refits, shared score matrix, and each has a one-number
   falsification test on adult. §5
9. **Selective editing (official statistics, 1980s–) already does cost-weighted
   triage** — suspicion × impact, thresholded. Supports our framing *and* is a
   prior-art risk R1 missed. Must be cited. §2.6

---

## 1. Is this a named problem? Yes — four names, in four communities

Our failure mode has been studied under distinct vocabularies that barely cite
each other. Using the right name matters, because one of them (official
statistics) has a 50-year head start.

### 1.1 "Error localisation" — official statistics / statistical data editing

**This is the oldest and most exact name for our problem.** It is defined as:
given a record that fails some edit rules, decide *which fields* to treat as
erroneous — explicitly separated from imputation, which then supplies values.

- Fellegi, I.P. and Holt, D. **A Systematic Approach to Automatic Edit and
  Imputation.** *Journal of the American Statistical Association* 71(353),
  1976. The founding formulation: change the *smallest number of fields* so
  the record satisfies all edits.
  Formulation and its generalisations confirmed in: Scholtus, S.
  **A generalized Fellegi–Holt paradigm for automatic error localization.**
  Statistics Canada *Survey Methodology* 12-001-X, 2016,
  <https://www150.statcan.gc.ca/n1/pub/12-001-x/2016001/article/14538/01-eng.htm>
- The **generalised** paradigm replaces "smallest number of fields" with
  "smallest sum of **reliability weights**" — non-negative per-variable numbers
  expressing how much one trusts each field's value. *That is literally a
  hand-set per-column corruption prior.* (same source)
- de Waal, T. **Solving the error localization problem by means of vertex
  generation.** *Survey Methodology* 29(1), 2003,
  <https://www150.statcan.gc.ca/n1/pub/12-001-x/2003001/article/6609-eng.pdf>
- Practical solvers state it as a mixed-integer program; agency systems include
  SPEER (US Census Bureau) and SLICE (Statistics Netherlands). MIP formulation:
  <https://www.cbs.nl/-/media/_pdf/2018/30/dp-errorloc-mip-v9.pdf>
- Runnable implementation: R package **`errorlocate`** (Edwin de Jonge), which
  compiles edit rules into a MIP and returns a minimum-weight set of fields to
  change. <https://data-cleaning.r-universe.dev/errorlocate/doc/manual.html>

**Why it matters, and its hard limit.** Fellegi–Holt localisation is *entirely*
driven by edit rules: with no violated edit there is nothing to localise.
Adult's corruption violates no edit (a valid `income` value replaced by the
other valid `income` value), so classical error localisation returns the empty
set on every row. **Its contribution to us is the vocabulary and the
reliability-weight idea, not the algorithm.**

### 1.2 "Cellwise outlier detection" — robust statistics

Robust statistics reframed outlier detection from *rows* to *cells* precisely
because most entries in a row are regular and only a few are anomalous.

- Rousseeuw, P.J. and Van den Bossche, W. **Detecting Deviating Data Cells.**
  *Technometrics* 60(2), 2018. arXiv: <https://arxiv.org/abs/1601.07251>.
  Method **DetectDeviatingCells (DDC)**: predict each cell from the *other*
  variables, standardise the residual robustly, flag cells whose robust
  standardised residual exceeds a cutoff; flag a *row* when the number and size
  of its flagged cells exceed a second cutoff. Presented as the first approach
  to find deviating cells while accounting for correlations among variables. It
  also returns predicted values for flagged cells and imputes missing ones.
- Software: R package **`cellWise`** (CRAN), acknowledged in the paper.

**Why it matters.** DDC *is* our conditional model, built by statisticians in
2018, with two ingredients we lack: (a) **robust** scale estimation, so the
residual threshold is not inflated by the contamination itself, and (b) an
explicit aggregation step from cells to a larger unit (there: the row). Its
per-cell test is still per-cell and it does not rank columns. But it is the
correct prior art for "residuals from a conditional model", and `cellWise` is a
runnable baseline (§4).

### 1.3 "Error diagnosis / attribution" — databases

The DB community asks the closest question: given observed errors, *which
attributes or value-groups explain them*?

- Wang, X., Dong, X.L., Meliou, A. **Data X-Ray: A Diagnostic Tool for Data
  Errors.** *SIGMOD 2015.* <https://lunadong.com/publication/dataXray_sigmod.pdf>
  Argues errors are **systematic**, so cell-by-cell cleaning does not stop
  recurrence; instead it finds *groupings of errors that share a cause*, scored
  by a **Bayesian cost model**, searched top-down, parallelisable.
  (Correction to my own prior expectation: the scoring is Bayesian, **not** MDL.
  I found no MDL formulation in the retrieved material.)
  Demo: **Error Diagnosis and Data Profiling with Data X-Ray**, VLDB 2015,
  <https://lunadong.com/publication/dataXray_vldbdemo.pdf>

**Why it matters.** Data X-Ray is the closest published statement of
*attribute-level error attribution*, and it is explicitly a **diagnosis** tool,
not a detector — the authors' own analogy is that an X-ray supports a diagnosis
without being one. Its input is *a known set of errors*. It answers "which
column do my errors live in?" only once you have errors. We need the
unsupervised version, which is §2.

### 1.4 "Column-level data validation" — production data-quality systems

- Song, J. and He, Y. **Auto-Validate: Unsupervised Data Validation Using
  Data-Domain Patterns Inferred from Data Lakes.** *SIGMOD 2021*, DOI
  10.1145/3448016.3457250. <https://arxiv.org/abs/2104.04659>
  The unit of validation is **the column**: infer a pattern for a column by
  borrowing from *similar columns elsewhere in a data lake*, so the pattern's
  false-positive rate can be estimated before deployment. Stated motivation:
  existing validators (Google's and Amazon's) produced **many false positives
  on string-valued production data**. Ships in part as Auto-Tag in Microsoft
  Azure Purview. Benchmark: <https://github.com/jiesongk/auto-validate>

**Why it matters.** This is the one published line whose *unit of decision is
the column*, and its central trick is the one we lack: **judge a column-level
rule using evidence from outside that column** (a corpus of other columns). Our
analogue of a data lake would be the other 14 columns, or synthetic corruption
(§2.3).

### 1.5 Naming recommendation

Use **"column-level error localisation"**, citing Fellegi–Holt for the term
"error localisation", Rousseeuw & Van den Bossche for the cellwise framing, and
Data X-Ray for attribution. Do **not** say "column anomaly detection" — that
phrase belongs to schema/drift monitoring (TFDV, Deequ) and will be read as
distribution-shift monitoring.

---

## 2. What signal separates "unusual values" from "tampered column"?

This is the crux. Each family is reported as: **who did it, on what data, what it
bought, does it need labels** — plus a verdict on whether the *column-level*
version exists in the literature or whether we would be first.

**The single most important finding in this section.** A "predictability shift"
signal is only meaningful against a **reference for what the column's
predictability *should* be**. The literature supplies exactly three ways to get
that reference:

| Source of reference | Published exemplar |
|---|---|
| A corpus of *similar columns elsewhere* | Auto-Validate (SIGMOD 2021) |
| A corpus of *other tables* + hypothesis tests | Uni-Detect (SIGMOD 2019) |
| *Synthetic corruption* injected by us | HoloDetect (SIGMOD 2019), Jenga (EDBT 2021) |

Nothing else in what I read provides a reference distribution. **A column-level
statistic with no reference is not a triage signal — it is just a number.** This
is the design constraint §5 is built around.

### 2.1 Predictability shift (per-column cross-validated predictability)

**Verdict — CORRECTED after reading ED2's technical report. This has been done,
in exactly the form the brief proposes, and it is the single most important
prior art for W4.5. See §2.1b. Do not claim novelty for per-column
predictability-based column selection.**

The *unsupervised* version (no labels at all) is still `not found`.

| Who | Data | What it bought | Labels? |
|---|---|---|---|
| Jäger & Bießmann, **Conformal Data Cleaning**, AISTATS 2024, PMLR v238 ([PDF](https://proceedings.mlr.press/v238/jager24a/jager24a.pdf)) | 16 OpenML datasets | One AutoGluon model **per column**, conformal prediction set per cell; beats baselines in ~61 % of experiments (R1 §4.4) | No error labels; needs a clean-ish training split |
| Rousseeuw & Van den Bossche, **DDC**, *Technometrics* 60(2), 2018 ([arXiv](https://arxiv.org/abs/1601.07251)) | numeric/mixed tables | Per-cell robust standardised residual from the other variables; aggregates cells to **row** flags | None |
| Deequ, per the open-source DQ tool survey ([arXiv 2407.18649](https://arxiv.org/pdf/2407.18649)) | production tables | Ships a **mutual-information check whose input is two columns and whose output is pass/fail** | None (threshold is config) |

CDC builds precisely the per-column models we would need and then **discards the
column-level information**, consuming each model only through a per-cell
prediction set. The only system I found that ranks columns by an aggregate of
per-column model quality is ED2 (§2.1b), whose *Max Error* criterion is exactly
per-column cross-validated F1 — and it reports that criterion as unstable.

The reference problem bites immediately: on Adult, `income` is intrinsically
only ~85–87 % predictable (*derived* from the published Adult classification
ceiling, R1 §R-1) while `native_country` is far *more* predictable, being nearly
constant. So raw predictability is confounded by the column's own entropy and is
not comparable across columns without normalisation — which is a plausible
explanation for ED2's reported instability, though they do not say so.

**Related but not the same:** per-feature two-sample testing for dataset shift.
Rabanser, Günnemann & Lipton, **Failing Loudly: An Empirical Study of Methods
for Detecting Dataset Shift**, NeurIPS 2019
([PDF](https://arxiv.org/pdf/1810.11953), code
<https://github.com/steverab/failing-loudly>) evaluate **univariate two-sample
tests per feature aggregated with Bonferroni** against multivariate tests. A
genuine column-level statistic — but it needs a *reference sample* of the same
column, which for us would have to be synthetic. The recipe was adopted
downstream, e.g. AWS Renate's KS detector, documented as "a KS test separately
for each feature, aggregated with a Bonferroni correction, as proposed in
Rabanser et al."
(<https://renate.readthedocs.io/en/dev/_apidoc/renate.shift.ks_detector.html>).

### 2.1b ED2's Column Selector — the prior art that matters most

**Neutatz, F., Mahdavi, M., Abedjan, Z. "ED2: A Case for Active Learning in
Error Detection", CIKM 2019** ([anthology entry](https://ir.webis.de/anthology/2019.cikm_conference-2019.253));
technical report **"ED2: Two-stage Active Learning for Error Detection"**,
[arXiv 1908.06309](https://arxiv.org/pdf/1908.06309) (HTML:
<https://ar5iv.labs.arxiv.org/html/1908.06309>); code
<https://github.com/BigDaMa/ExampleDrivenErrorDetection>.

ED2 contains a component literally called the **Column Selector**, which decides
**which column to label next**. From the technical report:

- **Architecture.** One XGBoost classifier **per column**. Stage 1 (the "column
  dimension") picks a column; stage 2 (the "cell dimension") picks cells inside
  it by **query-by-committee** in batches of k = 10, preferring distinct values
  for diversity. Initialisation uses a frequency-based ranker that proposes a mix
  of rare and frequent values per column, and needs **at least two erroneous and
  two correct cells per column** before it can start. One round-robin pass runs
  first, to collect the metrics the selector needs.
- **The four column-selection criteria they tried**, which map almost
  one-to-one onto the brief's signal families:
  | ED2 criterion | What it computes | Our §2 family |
  |---|---|---|
  | **Min Certainty** (default) | lowest **average classifier certainty** over the column's cells | confidence aggregation |
  | **Max Error** | lowest **average cross-validation F1** for that column's classifier | **predictability shift — exactly the brief's family (a)** |
  | Max Prediction Change | largest change in predictions between iterations | disagreement/stability |
  | Random / Round-Robin | — | uniform allocation baseline |
- **Their result.** **Min Certainty is reported as the most stable.**
  **Max Error and Max Prediction Change "can get stuck in local optima."**
- **Why they say column choice matters.** Columns converge at different speeds —
  their example is a `City` column converging after two iterations while a
  `Middle Name` column keeps consuming labels — and the chosen column's
  predictions also feed the **error-correlation features** of the other columns
  (each column's model receives the other columns' estimated error probabilities
  as features).
- **Imbalance is addressed explicitly**: "the class of erroneous values is much
  smaller than the class of correct values", so different columns need different
  label amounts.
- **Reported numbers** (Table 2, best F1): Flights (35 % errors) P 0.84 / R 0.90
  / F1 0.87 ± 0.01; Movies (1 % errors) P 0.89 / R 0.67 / F1 0.75 ± 0.12; Beers
  (8 % errors) P 1.00 / R 1.00 / F1 1.00 ± 0.00; Address (14 % errors) P 0.99 /
  R 0.96 / F1 0.97 ± 0.01. Abstract claim: "less than 1 % labels to outperform
  existing error detection approaches". Per-dataset label counts appear only in
  figures and are **not quantified in the text** — `not found` as numbers.
- **The honest warning for us**, in their own words about their lowest-prevalence
  dataset: on Movies (1 % errors) "ED2's results vary across runs" and Min
  Certainty's advantage is small "because error fractions are similar across its
  columns". **Our Adult is 0.25 % prevalence with error fractions that are
  maximally dissimilar across columns (one column has everything, fourteen have
  nothing) — the opposite regime, which is favourable for the mechanism but
  outside anything ED2 reports.**

**Consequences for the project, stated plainly.**

1. **Column-level triage is not a new idea.** It is a named component of a
   CIKM 2019 system that R1 listed only as "ED2 — active learning" and whose
   two-stage structure R1 missed. `docs/literature.md` §1 should be amended.
2. **Three of our proposed signals are ED2's rejected variants.** Per-column
   cross-validated predictability is their *Max Error*, and they report it
   unstable. Any W4.5 result must be compared against Min Certainty, Max Error,
   Round-Robin and Random, because those are the published baselines and we
   would otherwise again be beating only baselines we wrote ourselves (gap G3).
3. **What is still open.** ED2's selector is (a) **supervised** — it requires ≥2
   known errors *and* ≥2 known correct cells **per column** to initialise, which
   on Adult is impossible for the 14 uncorrupted columns and would therefore
   never start; (b) built on **classifier certainty**, i.e. a density-type
   quantity, not a likelihood ratio (§3.1) or a contamination fraction (§2.3);
   and (c) never evaluated at sub-1 % prevalence with corruption concentrated in
   one column. **The unsupervised, prevalence-estimating column prior is the
   remaining gap, and it is narrower than it looked an hour ago.**

### 2.2 Asymmetry / mutual information (A predicts B, B no longer predicts A)

**Verdict: `not found` in data cleaning. Do not claim it exists.**

Two targeted searches returned no method that identifies *which* variable is
corrupted from the pattern of broken dependencies. What does exist:

- **Deequ's two-column MI check** (above) detects weakened dependence but says
  nothing about which of the two columns caused it. That is exactly the
  asymmetry problem, unsolved.
- **Measurement error breaks conditional-independence tests, and the causal
  literature knows it.** **An Upper Bound for Random Measurement Error in Causal
  Discovery**, [arXiv 1810.07973](https://arxiv.org/pdf/1810.07973): under
  ordinary faithfulness, a conditional independence that appears because the
  conditioning set is noisy can only be explained by a faithfulness violation;
  the authors adopt a stronger assumption plus an error correction built on a
  noise upper bound. PC-ME (IJCAI 2025,
  <https://www.ijcai.org/proceedings/2025/1013.pdf>) handles the same problem
  with rank constraints on high-order cumulants.
- **Coherency scores.** Faltenbacher, Wahl, Herman, Runge, **How PC-based
  Methods Err: Towards Better Reporting of Assumption Violations and Small
  Sample Errors**, UAI 2026, PMLR v337
  ([paper](https://proceedings.mlr.press/v337/faltenbacher26a.html),
  [arXiv 2502.14719](https://arxiv.org/pdf/2502.14719)). They show *local* errors
  can corrupt edge orientations **far from the error source**, and propose
  coherency scores computed only from the CI tests the algorithm already ran.
  Their own caveat: some outputs are coherent and still wrong, and no check based
  only on the test list and output graph will catch those.
- Mislabel detection by **pointwise mutual information**, decomposing a global MI
  into per-row contributions ([arXiv 2508.07713](https://arxiv.org/pdf/2508.07713)) —
  but on **MNIST with synthetic label noise**, input-vs-label rather than
  column-vs-column, and per-*row* rather than per-column. A direction, not a
  method.

**Honest read:** asymmetry is the most interesting idea in the brief and the one
with the least published support. Faltenbacher et al.'s "local error, non-local
damage" result is in fact a **warning against it**: damage from a corrupted
column does not stay near that column, so attributing broken dependencies to a
column is unsound without further assumptions.

### 2.3 Self-supervised corruption injection, per column

**Verdict: per-column injection exists (for a different purpose); the specific
"which column's detector responds *least*" test is `not found`.**

| Who | Data | What it bought | Labels? |
|---|---|---|---|
| Heidari et al., **HoloDetect**, SIGMOD 2019 | 5 datasets incl. Adult | Learned augmentation policies: **+20 F1 average, 3x fewer labels**, beats active learning on recall **in all cases** (R1 §3.3) | Weak supervision from the dirty data |
| Schelter, Rukat, Bießmann, **JENGA — A Framework to Study the Impact of Data Errors on the Predictions of Machine Learning Models**, EDBT 2021 ([PDF](https://cdn.amazon.science/ed/04/f752caff482788a214dfb7460c43/jenga-a-framework-to-study-the-impact-of-data-errors-on-the-predictions-of-machine-learning-models.pdf), [PyPI](https://pypi.org/project/jenga/)) | tabular prediction tasks | An extensible library of **synthetic corruptions (missing values, outliers, typos, noisy measurements)** plus evaluation, used to stress-test models and **TFDV integrity constraints** | None |
| Liu, Zhou, Rekatsinas, **Picket**, VLDB J 31(5), 2022 | small tabular | Self-supervised reconstruction loss at training *and* inference time | None |
| CDC (above) | 16 OpenML datasets | Corruption suite includes **column swaps**, scaling, Gaussian noise, categorical shift at 1/5/10/30/50 % | None |

Whether Jenga's corruptions are applied **per named column** is `not found` in
the material I could retrieve — the abstract lists corruption *types*, not their
scope. **Inspect `jenga`'s corruption classes before citing it as per-column.**

**The missing piece, and the machinery that supplies it.** The brief's idea —
inject corruption into each column and see whose detector responds *less*
because its baseline is already polluted — is formally **estimating each
column's contamination fraction**. That is a solved problem with a name:

- **Mixture Proportion Estimation (MPE).** Garg, Wu, Smola, Balakrishnan,
  Lipton, **Mixture Proportion Estimation and PU Learning: A Modern Approach**,
  NeurIPS 2021
  ([proceedings](https://papers.nips.cc/paper/2021/hash/47b4f1bfdf6d298682e610ad74b37dca-Abstract.html)).
  Splits PU learning into (i) estimating the fraction of positives in an
  unlabelled pool and (ii) training a classifier. Contributes **BBE (Best Bin
  Estimation)** with formal guarantees "whenever we can train a model to cleanly
  separate out a small subset of positive examples", plus CVIR, alternated in a
  loop the authors call (TED)^n. Explicitly motivated by classical MPE methods
  breaking down in high dimensions and by heuristics that "depend precariously
  on hyperparameter tuning".
- **Anomaly detection with contaminated training data.** Qiu et al., **Latent
  Outlier Exposure for Anomaly Detection with Contaminated Data**, ICML 2022,
  PMLR v162:18153–18167 ([PDF](https://arxiv.org/pdf/2202.08088)). Jointly infers
  binary normal/anomalous labels while updating model parameters; evaluated on 3
  image datasets, **30 tabular datasets** and a video benchmark. Its related-work
  names the alternative "Refine" (iteratively weeding anomalies out of the
  training set, after Yoon et al.; the LOE paper dates that 2021 and a citing
  paper 2022 — check before citing). A citing paper reports LOE still works when
  the anomaly ratio is mis-specified within 5 percentage points —
  **[UNVERIFIED]**, secondary source, not the LOE paper. See also **Impact of
  Inaccurate Contamination Ratio on Robust Unsupervised Anomaly Detection**
  ([arXiv 2408.07718](https://arxiv.org/html/2408.07718v1)).

**This is the strongest lead in the document.** "Which column is tampered with"
= "which column has the highest contamination fraction", and MPE estimates
contamination fractions **without cell labels**. It reframes column triage from
an anomaly-*scoring* problem (where rarity confounds) into an **estimation**
problem (where it does not: a uniformly-rare-but-clean column has a *low*
mixture weight on the anomalous component, because its rare values are not
separable from its own distribution by any trained model).

### 2.4 Two-sample / distributional tests vs the model-implied conditional

**Verdict: the machinery is standard and well cited; its use as a column-level
corruption test in data cleaning is `not found`.**

- **The exact right residual for this.** Dunn, P.K. & Smyth, G.K. **Randomized
  Quantile Residuals**, *Journal of Computational and Graphical Statistics*
  5(3):236–244, 1996
  ([record](https://research.usc.edu.au/esploro/outputs/journalArticle/Randomized-quantile-residuals/99449465702621)).
  Invert the fitted CDF at each observed response and map through the standard
  normal quantile function; residuals are **exactly normal** apart from sampling
  variability in the estimated parameters. For **discrete** responses, randomise
  within the probability interval the jump creates. Implemented in R `statmod`
  (`qresiduals`, <https://rdrr.io/cran/statmod/man/qresiduals.html>) and
  `topmodels::proresiduals`
  (<https://topmodels.r-forge.r-project.org/reference/proresiduals.html>).
  **Why it matters to us:** our columns are mostly categorical, and this is the
  published, correct way to turn a *categorical* conditional predictive
  distribution into a residual that is uniform/normal **under the null that the
  model is right**. A one-sample goodness-of-fit test on a column's residuals is
  then a direct test of "is this column consistent with its own conditional
  model". The `statmod` docs note quantile residuals are "the only useful
  residuals for binomial or Poisson data when the response takes on only a small
  number of distinct values" — our categorical case. Later work proves the
  uniformity Dunn & Smyth did not
  ([arXiv 1708.08527](https://www.arxiv.org/pdf/1708.08527)), reports RQRs have
  "low type I error and great statistical power" for detecting misspecification
  in count regression, and records that RQRs have **not** been widely adopted as
  a standard diagnostic — so this is defensible and uncrowded ground.
- **Classifier two-sample test.** Lopez-Paz & Oquab, **Revisiting Classifier
  Two-Sample Tests**, ICLR 2017
  ([arXiv 1610.06545](https://arxiv.org/abs/1610.06545)). Label P samples
  positive and Q negative; under H0: P = Q held-out accuracy stays near chance,
  so above-chance accuracy is evidence of a difference. Accuracy is an average of
  independent variables and hence has an asymptotically Gaussian null — a simple
  null distribution, in interpretable units. The authors also note the
  classifier's predictive uncertainty can indicate **where** P and Q differ.
  **Read-across:** train one classifier per column to separate that column's
  *observed* values from values *resampled from its own model-implied
  conditional*. Above-chance accuracy means the column's observed distribution is
  not what its conditional model implies. This is the two-sample form of the
  triage signal, with a p-value attached.
- **Per-feature KS + Bonferroni** — Failing Loudly, §2.1.
- **Production practice:** Auto-Validate's entire contribution is estimating a
  column-level rule's false-positive rate *before* deploying it (§1.4).

### 2.5 Residual concentration (few rows vs spread thin)

**Verdict: published machinery exists and is directly reusable; nobody has
applied it per column for error localisation.**

- **Fit a two-component mixture to the score distribution and read off the
  weight.** Gao, J. & Tan, P-N. **Converting Output Scores from Outlier Detection
  Algorithms into Probability Estimates**, ICDM 2006
  ([PDF](https://cse.buffalo.edu/~jing/doc/ICDM06.pdf)). Two transformations: a
  **sigmoid** fit (posteriors assumed logistic in the score) and a **mixture**
  fit, modelled as a **mixture of exponential and Gaussian** components with
  posteriors from Bayes' rule. Stated motivations: choose a threshold via a
  **Bayesian risk model**, and aggregate models into an ensemble. Calibration
  accuracy improves given some labelled examples.
  **This is the concentration measure the brief asks for:** the fitted mixture
  weight on the anomalous component *is* an estimate of the anomalous fraction,
  and it distinguishes "a few extreme rows" from "a thin uniform tail" by
  construction. Runnable: `pythresh.thresholds.mixmod`
  (<https://pythresh.readthedocs.io/en/latest/_modules/pythresh/thresholds/mixmod.html>).
  Note it is also the *exact same quantity* as the MPE estimate in §2.3, reached
  by a different route — two independent literatures converging on
  "estimate the mixture weight" is a good sign.
- **Label-free evaluation of a scoring.** Goix, N. **How to Evaluate the Quality
  of Unsupervised Anomaly Detection Algorithms?**
  ([arXiv 1607.01152](https://arxiv.org/abs/1607.01152)) — Excess-Mass and
  Mass-Volume criteria need **no labels** and are reported to discriminate
  between algorithms about as well as ROC/PR criteria that do. The paper's own
  caveat: EM/MV "generally cannot be well estimated in large dimension",
  addressed by feature sub-sampling and aggregating. Per *column* we are in one
  dimension, the regime where these criteria are strongest.
- **Clumped anomalies need special handling.** **A Large-scale Study on
  Unsupervised Outlier Model Selection: Do Internal Strategies Suffice?**
  ([arXiv 2104.01422](https://arxiv.org/pdf/2104.01422)) records that collective
  outliers forming micro-clusters do **not** have high separability, and that
  IREOS handles this given a user-specified `clump_size`. Directly relevant: a
  tampered column's corruptions *are* a clump.
- Alternative route: **MetaOD** — Automatic Unsupervised Outlier Model Selection,
  NeurIPS 2021 (<https://neurips.cc/virtual/2021/poster/28754>) — selects a model
  with no labels, no model evaluations and no universal objective, by
  meta-learning from historical benchmarks; selection from 300+ models in < 1 s.

### 2.6 Spending a handful of labels on COLUMNS rather than cells

**Verdict: the formal machinery exists outside data cleaning and this is the
best-supported recommendation in the document.**

- **Selective editing (significance editing), official statistics.** The standard
  procedure is exactly a triage: a **local score per variable**, combined into a
  **global score per unit**, a **threshold**, and units above it sent to
  interactive (human) editing. Key references named in the methodological
  literature: Hidiroglou & Berthelot (1986), Latouche & Berthelot (1992),
  Lawrence & McKenzie (2000), Hedlin (2003), Norberg (2016). Overview: **The
  Unknown Future of Statistical Data Editing: Some Imputations**, SAGE /
  *Statistical Journal of the IAOS*, 2025
  (<https://journals.sagepub.com/doi/10.1177/0282423X251333718>). Applied
  example: **Score Functions for Selective Editing of the U.S. Census Bureau
  Trade Data**, Census RRS 2012-10
  (<https://www.census.gov/library/working-papers/2012/adrm/rrs2012-10.html>),
  which splits the score into **how suspicious a record is** times **how much
  changing it would move publication totals**.
  Reported efficiency, all quoted from the SAGE overview (which attributes them
  to the underlying studies — **cite via that overview, not as primary**):
  ~**5 % of erroneous units accounted for 90 % of published estimates** in one
  economic survey; the **top-ranked 20 % of erroneous records made up 90 % of
  the total adjustment** in another; Granquist & Kovar reported **cost savings of
  50 % or more** with only a small effect on final estimates. Editing is put at
  **up to 20 % of production resources**, and the literature names
  **over-editing** as a recognised failure mode. Stated caveats: selective
  editing "has no single accepted theory"; it assumes true values for flagged
  units can be obtained by follow-up.
  **This is the closest analogue to our cost-sensitive decision layer in any
  mature production discipline, and R1 did not find it. Cite it in the paper
  regardless of what W4.5 implements** — it is simultaneously supporting
  evidence for our framing and a prior-art risk we must address head-on.
- **Active search** — maximise the number of positives found under a query
  budget. Jiang, Malkomes, Converse, Shofner, Moseley, Garnett, **Efficient
  Nonmyopic Active Search**, ICML 2017, PMLR 70
  ([ICML page](https://icml.cc/virtual/2017/talk/1176); extended summary
  [arXiv 1811.08871](https://arxiv.org/pdf/1811.08871)). Proves **no
  polynomial-time policy achieves a constant-factor approximation** to the
  optimal policy's expected utility; contributes an efficient nonmyopic policy
  that balances exploration and exploitation using the whole remaining budget
  with **no tuning parameter**, recovering significantly more positives at equal
  budget on drug-discovery, materials-science and citation data. Multiclass
  extension with diminishing returns:
  [arXiv 2202.03593](https://arxiv.org/abs/2202.03593v3).
  **Read-across:** at 0.25 % prevalence, "find corrupted cells with 20 labels"
  *is* active search, not active learning — the objective is positives found,
  not classifier accuracy. Our W3 wave optimised the wrong objective.
- **Rare-category detection.** Pelleg & Moore, **Active Learning for Anomaly and
  Rare-Category Detection**, NIPS 2004, pp. 1073–1080
  ([PDF](https://papers.neurips.cc/paper_files/paper/2004/file/8c59fd6fbe0e9793ec2b27971221cace-Paper.pdf)).
  Fits a mixture model with no assumption on component form; reports it "can
  quickly zoom in on an anomaly set containing a few tens of points in a dataset
  of hundreds of thousands". Extended by He & Carbonell, **Nearest-Neighbor-Based
  Active Learning for Rare Category Detection**, NIPS 2007
  (<https://proceedings.neurips.cc/paper/2007/hash/2838023a778dfaecdc212708f721b788-Abstract.html>),
  which describes Pelleg & Moore as the prior best technique in a sparse
  literature.
- **How to split a budget across groups, formally.** Neyman allocation is the
  textbook answer and recent work makes it adaptive: **A Complexity Measure for
  Active Learning in Multi-group Mean Estimation**, PMLR v336, 2026
  (<https://proceedings.mlr.press/v336/aznag26a.html>) — a learner adaptively
  allocates a budget of T samples across d groups to minimise the worst-case
  per-group uncertainty index max_k (sigma_k^2 / n_k), with a lower bound
  separating a budget term, a heteroscedasticity index and a curvature term.
  **Near-Exponential Savings for Mean Estimation with Active Learning**
  ([arXiv 2511.05736](https://arxiv.org/html/2511.05736v1)) states the ideal
  strategy — partition into strata minimising within-stratum variance, then
  allocate across strata by **Neyman allocation** — and notes the optimal
  stratification is usually unknown in advance; implemented in the R package
  `PartiBandits`. Fixed baselines worth copying from **Benchmarking Multi-Domain
  Active Learning on Image Classification**
  ([arXiv 2312.00364](https://arxiv.org/pdf/2312.00364)): uniform split, split
  proportional to error rate, and all-budget-to-the-worst-group.
  **Read-across: columns are the strata.** Our W3 finding that only
  budget-spreading strategies learned anything is the *uniform allocation*
  baseline; Neyman allocation is its principled version.

### 2.7 Other things the literature actually uses

- **A hand-set per-column corruption prior**: the generalised Fellegi–Holt
  **reliability weights** (§1.1). Crude, 1976, still deployed by national
  statistical agencies. It is the honest baseline any learned column prior must
  beat, and it is free to implement.
- **Corpus-calibrated hypothesis testing.** Wang, P. & He, Y. **Uni-Detect: A
  Unified Approach to Automated Error Detection in Tables**, SIGMOD 2019,
  pp. 811–828, DOI 10.1145/3299869.3319855
  ([PDF](https://microsoft.com/en-us/research/uploads/prod/2019/04/Uni-Detect.pdf)).
  Performs **local data perturbations ("what-if" analysis) to reason about data
  abnormality, leveraging classical hypothesis tests on a large corpus of
  tables**; reports finding thousands of FD violations, numeric outliers and
  spelling mistakes in Wikipedia and other tables with better accuracy than
  algorithms designed per error type, and beating a commercial search engine's
  spell-checker on spelling. Explicit motivation: removing per-dataset
  configuration (FDs between column pairs, tuned outlier thresholds).
  **The PDF body is FlateDecode-compressed and I could not extract the null
  hypothesis, the statistic, the corpus-calibration procedure, or any
  precision/recall figures.** The related patent (US 11157479,
  <https://patents.justia.com/patent/20200320053>) describes removing small
  subsets of rows and comparing the perturbed table's probability against the
  original under a null that the table resembles the training tables — but that
  is a separate document and may not match the paper's algorithm.
  **[UNVERIFIED — read the paper body before citing the mechanism.]**
- **Attribution given known errors**: Data X-Ray (§1.3).
- **Triage by downstream impact rather than by suspicion.** Krishnan, Wang, Wu,
  Franklin, Goldberg, **ActiveClean: Interactive Data Cleaning For Statistical
  Modeling**, PVLDB 9, 2016 ([PDF](https://www.vldb.org/pvldb/vol9/p948-krishnan.pdf)) —
  treats clean-and-retrain as SGD and prioritises records "most likely to be
  dirty **and** most likely to change the results"; reports accuracy improvements
  **up to 2.5x** for the same cleaning budget, beating uniform sampling and
  active learning on all real dirty datasets, with the advantage largest **when
  errors are sparse and budgets are small** — our regime exactly.
  Krishnan, Franklin, Goldberg, Wu, **BoostClean: Automated Error Detection and
  Repair for Machine Learning**, [arXiv 1711.01299](https://arxiv.org/pdf/1711.01299) —
  targets **domain value violations**, uses clean test labels to judge whether a
  detect+repair combination improves model accuracy, and selects an ensemble by
  **statistical boosting**; up to **9 % absolute accuracy gain** over the best
  non-ensembled alternative across 12 datasets, with a 22.2x speedup from its
  optimisations. **Correction to a claim I expected to confirm and could not:
  BoostClean does not use a multi-armed bandit.** No source supports that; its
  selector is boosting, and its detector generates predicates via isolation
  forests on featurised values (per course slides,
  <https://classes.engr.oregonstate.edu/eecs/spring2019/cs519-001/BoostC.pdf>).
  Do not write "bandit" next to BoostClean.

---

## 3. Why rarity is not corruption, formally

### 3.1 Density is the wrong statistic; a likelihood ratio is the right one

The cleanest published formulation is not in data cleaning but in OOD detection.

Ren, J. et al. **Likelihood Ratios for Out-of-Distribution Detection**,
NeurIPS 2019
([proceedings](https://proceedings.neurips.cc/paper/2019/hash/1e79596878b2320cac26dd792a6c51c9-Abstract.html),
[arXiv 1906.02845](https://arxiv.org/abs/1906.02845)). The finding: a generative
model's **likelihood score is heavily affected by population-level background
statistics**, so low likelihood is not a reliable OOD signal. The fix is a
**likelihood ratio** between an in-distribution model and a **background model
trained on perturbed in-distribution data** (the perturbation amount is a tuned
hyperparameter), which "effectively corrects for these confounding background
statistics". Motivating application: bacterial identification from genomic
sequences, with a released genomics OOD benchmark. Honest caveats: the authors'
own blog post states the method reaches state of the art on the genomic dataset
but "does not yet have high enough accuracy to reach the standards for
deployment to real applications"
(<https://ai.googleblog.com/2019/12/improving-out-of-distribution-detection.html>);
a reviewer challenged the independence assumption behind the
background/semantic decomposition
(<https://papers.nips.cc/paper/2019/file/1e79596878b2320cac26dd792a6c51c9-Reviews.html>).

**This is our `native_country` failure, named and solved in another field.**
"United-States" is background frequency. A rare country is low-density under the
*clean* conditional model — but it is **equally low-density under any corruption
model**, so the likelihood *ratio* is near 1 and the cell is correctly not
flagged. Our detector scores `1 - P(observed | rest of row)`, which is a
**density**, exactly the statistic Ren et al. show is confounded. Replacing it
with a ratio against an explicit corruption model is the formally correct repair,
and it is citable.

Derived corollary, stated as derivation not measurement: by Bayes,
`P(corrupt | x) / P(clean | x) = [P(x | corrupt) / P(x | clean)] * [pi_c / (1 - pi_c)]`
where `pi_c` is the column's corruption prevalence. **Both factors are
column-level quantities we currently ignore**: the likelihood ratio needs a
corruption model (§2.3, §3.1) and the prior odds need a prevalence estimate
(§2.3 MPE, §2.5 mixture weight). Column triage is not a heuristic add-on — it is
the two missing terms of the correct per-cell posterior.

### 3.2 The taxonomy: anomaly vs outlier vs novelty

Ruff, L. et al. **A Unifying Review of Deep and Shallow Anomaly Detection**,
*Proceedings of the IEEE* 109(5):756–795, 2021, DOI 10.1109/JPROC.2021.3052449
([arXiv 2009.11732](https://arxiv.org/abs/2009.11732)). Verbatim definitions
(§II-B of the paper):

- **Anomaly**, general: "An anomaly is an observation that deviates considerably
  from some concept of normality" — formally a point in the low-density set
  A = {x : p+(x) <= tau}.
- **Anomaly**, strict: "an anomaly is often characterized as being an instance
  from a distinct distribution other than" P+.
- **Outlier**: "a rare or low-probability instance from" P+.
- **Novelty**: "an instance from some new region or mode of an evolving,
  non-stationary" P+.
- The review then states "we make no distinction between these terms".

**That last sentence is the gap we can stand in.** The *strict anomaly* (a draw
from a different distribution — our corruption) and the *outlier* (a rare draw
from the same distribution — `native_country = Holand-Netherlands`) are formally
different objects, and the field's own unifying review declines to separate them.
Also quotable from the same paper:

- "Anomalous data may be useless, for example when caused by measurement errors."
- Outliers "are frequently regarded as 'noise' or 'measurement error'" to be
  removed in preprocessing, "whereas anomalies are often the data points of
  interest".
- The **concentration assumption**: "a fundamental assumption in anomaly
  detection is that the region where the normal data lives can be bounded" —
  and critically, "the set of anomalies A **need not be concentrated** and can be
  unbounded".
- "semantic anomalies can be very close to normal instances in the raw feature
  space."
- On scoring, methods target "some strictly increasing transformation" of the
  density, which "is usually no longer a proper density but still preserves the
  density ranking", e.g. negative log-likelihood. The review **does not** discuss
  likelihood ratios, which confirms §3.1 is not standard practice in that field
  either.

Independent confirmation that rarity is not the target, from 2004: Pelleg & Moore
(§2.6) build their method around "useful" anomalies decided subjectively by the
user, explicitly "set apart from the classic statistical view of anomalies as
outliers or poorly modeled points", and name their uninteresting rare categories
("boring" anomalies: diffraction spikes, a known optical artefact, and
satellites overflying during the exposure).

### 3.3 How deployed systems avoid flagging legitimately rare values

Four distinct published mechanisms, none of which we currently use:

1. **Estimate the rule's false-positive rate on outside evidence before
   deploying it.** Auto-Validate (§1.4) infers a column's pattern from *similar
   columns in a data lake* precisely because existing validators "produced many
   false positives on string-valued production data". Shipped as Auto-Tag in
   Azure Purview.
2. **Calibrate significance against a corpus of tables.** Uni-Detect (§2.7).
   **[mechanism UNVERIFIED — see the extraction failure noted there.]**
3. **Require the error to matter.** Selective editing (§2.6) weights
   suspiciousness by **impact on published estimates**, so a rare-but-harmless
   value is never followed up. The US Census trade score is literally
   suspicion times influence.
4. **Require it to cost something.** REIN's asymmetry finding (R1 §4.2) —
   precision matters more for repair quality, but false negatives become more
   harmful once the repairer is strong — is the published statement that the
   operating point must depend on costs. Our Phase 7–8 layer *is* this
   mechanism; §3.1 explains why it is currently fed a mis-specified `q`.

**Negative result worth recording.** I found **no** published system that
separates rare-but-valid from corrupted *using only the single table under
inspection and no labels*. Every mechanism above imports outside information: a
corpus, a label budget, an impact model, or a cost model. That is a hard
constraint on what W4.5 can achieve, and stating it in the paper is better than
discovering it during implementation.

---

## 4. The novelty gap (G3): which published system can we actually run?

Answering the brief's sub-questions first.

**Does REIN ship runnable implementations? YES.**
`https://github.com/mohamedyd/rein-benchmark` (last commit **2024-07-24**,
16 stars, 5 forks). Install: `python3 -m venv` then `pip3 install -e .`.
It needs **PostgreSQL** for its `rein` database (`create_db_user.sh`), but its
scripts **fall back to CSV files when the database lacks the data**. Detection
methods wrapped: `mvdetector, raha, ed2, max_entropy, min_k, metadata_driven,
holoclean, fahes, katara, openrefine, cleanlab, outlierdetector, zeroer,
duplicatesdetector, picket, nadeef` (its robustness example also uses `dboost`).
Per-method extra setup documented in its README: Raha/Baran `pip3 install raha`;
Picket `bash install.sh cpu`; FAHES `make` in `cleaners/FAHES/src/`; KATARA a
separate knowledge-base download; NADEEF needs FD rules and patterns; HoloClean
needs denial constraints plus its own Postgres user; ZeroER needs blocking
functions; OpenRefine needs clusters.
**Adding our data is documented and matches our layout almost exactly**: create
`datasets/<name>/`, put `dirty.csv` (and `clean.csv`) in it, then add a metadata
entry under `auxiliaries/` with paths, `error_types`, `ml_tasks`,
`excluded_attribs`, and the regression/classification label columns.

**Is "a previous attempt got Raha running" plausible? YES — fully plausible, and
here is the exact path.** Raha is a 2023 PyPI package whose benchmarking API
takes precisely a `{name, path, clean_path}` dictionary, which is our
`datasets/<name>/{dirty,clean}.csv` layout verbatim.

```bash
# in a SEPARATE venv (raha pins older libs; do not pollute the project env)
python -m venv .venv-raha && .venv-raha/Scripts/activate   # Windows
pip install raha                                            # PyPI 1.26, 2023-06-29
```

```python
import raha
dataset_dictionary = {
    "name": "breast_cancer",
    "path":       r"C:\...\datasets\breast_cancer\dirty.csv",
    "clean_path": r"C:\...\datasets\breast_cancer\clean.csv",
}
app = raha.Detection()          # LABELING_BUDGET = 20 by default
detection_dictionary = app.run(dataset_dictionary)   # -> {(row_idx, col_idx): "JUST A DUMMY VALUE"}
data = raha.Dataset(dataset_dictionary)
print(data.get_data_cleaning_evaluation(detection_dictionary)[:3])   # P, R, F1
```

Verified facts behind that snippet (all from the repo, not inferred):
- README installation is exactly `pip3 install raha`, or clone +
  `pip3 install -e raha`. Repo: <https://github.com/BigDaMa/raha>, **73 stars,
  181 commits, last commit 2025-06-05** ("Fix correction selection to use
  highest confidence instead of last…").
- PyPI `raha` **1.26**, uploaded **2023-06-29**, `requires_python` **not
  specified**, deps declared unpinned: `numpy, pandas, scipy, scikit-learn,
  matplotlib, prettytable, mwparserfromhell, beautifulsoup4, py7zr`.
  The repo's `requirements.txt` pins harder: `numpy==1.22.4, pandas==2.1.0,
  scipy==1.10.0, scikit-learn==1.2.2, matplotlib==3.7.1, prettytable==0.7.2,
  mwparserfromhell==0.5.4, beautifulsoup4==4.6.0, py7zr==0.20.2`.
  **Our project is on pandas 2.3; raha pins 2.1.0.** Hence the separate venv.
- `raha/detection.py`: class `Detection`, `run(dd)` returns
  `d.detected_cells`, a dict keyed by `(row_index, column_index)`.
  Hyperparameters: `LABELING_BUDGET = 20`, `USER_LABELING_ACCURACY = 1.0`,
  `CLUSTERING_BASED_SAMPLING = True`, `CLASSIFICATION_MODEL = "GBC"`,
  `LABEL_PROPAGATION_METHOD = "homogeneity"`,
  `ERROR_DETECTION_ALGORITHMS = ["OD", "PVD", "RVD", "KBVD"]`,
  `STRATEGY_FILTERING = False`, `HISTORICAL_DATASETS = []`.
  **Labelling only happens if `d.has_ground_truth` is true** — i.e. the 20
  labels are simulated from `clean.csv`, which is the accepted benchmarking
  practice for Raha/ED2/SAGED (R1 §R-3) and must be stated as such.
  Without ground truth the loop would not add labels and would not terminate.
- The paper won the **ACM SIGMOD Most Reproducible Paper Award**
  (<https://www.bifold.berlin/news-events/news/view/news-detail/researchers-in-prof-abedjans-group-win-simgod-reproducibility-award>),
  which is independent evidence that this is the right first baseline.
- Caveat for **Baran** (correction, not detection): the README warns the
  pre-trained model shipped in the repo is only a sample for the notebooks, and
  that experiments should train their own model from Wikipedia revision
  histories. Do not report Baran numbers using the shipped sample model.

### 4.1 Prioritised table — (likelihood of running) / (effort)

| # | System | Repo / package | Last activity | Stars | Install effort | Input format | Fits our clean/dirty CSV pairs? |
|---|---|---|---|---|---|---|---|
| **1** | **Raha** (detection) + Baran (repair), SIGMOD 2019 / PVLDB 2020 | [BigDaMa/raha](https://github.com/BigDaMa/raha), PyPI `raha` 1.26 | **2025-06-05** | 73 | **Trivial** — one `pip install` in a side venv | `{name, path, clean_path}` dict of CSV paths | **Yes, verbatim.** Zero adaptation. |
| **2** | **DDC / cellwise outliers**, *Technometrics* 2018 | PyPI **`robpy`** 0.0.6 (`robpy.outliers.DDC`) | **2025-09-09** | n/a | **Trivial** — `pip install robpy`, py>=3.10 | numeric matrix (DataFrame) | **Yes for breast_cancer / smartfactory numerics**; categoricals need encoding |
| **3** | **PyOD** (JMLR 2019) + `pythresh` for the §2.5 mixture weight | PyPI `pyod`, `pythresh` ([KulikDM/pythresh](https://github.com/KulikDM/pythresh)) | actively maintained | 9k+ / — | **Trivial** | numeric array | **Yes**, per column |
| **4** | **REIN harness** (unlocks min_k, max_entropy, metadata_driven, mvdetector, cleanlab, outlierdetector, + wrapped raha/ed2/dboost) | [mohamedyd/rein-benchmark](https://github.com/mohamedyd/rein-benchmark) | 2024-07-24 | 16 | **Medium** — venv + `pip3 install -e .` + PostgreSQL (CSV fallback exists) | `datasets/<name>/{dirty,clean}.csv` + metadata entry | **Yes, documented** |
| **5** | **dBoost** (the published best non-learning detector on Breast Cancer, F1 0.70 in SAGED) | [cpitclaudel/dBoost](https://github.com/cpitclaudel/dBoost), GPL-3.0 | **2015-12-03** | 19 | **Medium-high** — decade-dead Python, `./dboost-stdin.py -h`, input format undocumented | stdin tuples | Probably, after a shim; or take REIN's wrapper |
| **6** | **Conformal Data Cleaning**, AISTATS 2024 — *our main novelty risk* | [se-jaeger/conformal-data-cleaning](https://github.com/se-jaeger/conformal-data-cleaning) | 2 commits total | 24 | **High** | **`--task_id` (OpenML integer) only** | **No.** `run_experiment.py` has no file-path argument; data comes from `get_OpenMLTask(task_id)`. Needs a loader shim + AutoGluon + 50 HPO trials. Repo also carries a `garf/main.py` and a PyOD-ECOD baseline. |
| **7** | **ED2**, CIKM 2019 — *the key column-triage prior art* | [BigDaMa/ExampleDrivenErrorDetection](https://github.com/BigDaMa/ExampleDrivenErrorDetection) | 2021-06-01 | 12 | **High** — `sudo python setup.py install`, `libpq-dev/python-dev/python-tk` (looks Python 2), per-machine config files, run by editing `features_experiment_multi.py`, no license, minimal README | config file + `datasets/` folder | Awkward. **Better: reach ED2 through REIN, or reimplement its Column Selector criteria (they are 2–5 lines each).** |
| **8** | **HoloClean**, PVLDB 2017 | [HoloClean/holoclean](https://github.com/HoloClean/holoclean) | **2019-04-27** | — | **High** — PostgreSQL + PyTorch 0.4/1.0 era, and it needs **denial constraints** we would have to author (REIN: Adult F1 0.51 → 0.12 when rules drop 17 → 7) | DC file + CSV | Not recommended |
| **9** | **Picket**, VLDB J 2022 | [sis-ethz/Picket](https://github.com/sis-ethz/Picket) | **2020-11-24, single "initial commit"** | — | **High** — pins torch 1.0.1 / pandas 0.23.4 / sklearn 0.22.1 / gensim 3.8.1, notebook-driven demos | notebooks | **No.** REIN reports Picket **crashed with memory faults on Adult and Smart Factory** — our two large datasets. Do not spend effort here. |
| 10 | **errorlocate** (Fellegi–Holt MIP) | [R package](https://data-cleaning.r-universe.dev/errorlocate/doc/manual.html) | CRAN-active | — | Medium (R) | data + `validate` edit rules | Yes, but needs edit rules; returns the empty set on Adult by construction |
| — | **SAGED** (EDBT 2024) | code `not found` | — | — | — | — | — |
| — | **ZeroED** (2025), **ForestED** (2025) | code `not found` for both | — | — | — | — | — |

**Supporting tooling (not baselines, needed by §5).**
- **`jenga`** (EDBT 2021, PyPI `jenga`) — synthetic corruptions (missing values,
  outliers, typos, noisy measurements) + evaluation, from the authors of the
  corruption-robustness line of work.
- **`tab-err`** (PyPI **0.2.1, 2025-12-17**, [calgo-lab/tab_err](https://github.com/calgo-lab/tab_err),
  docs <https://tab-err.readthedocs.io/latest/>) — "Fully-controllable error
  generation for tabular data", which **separates the error *mechanism* (where
  errors occur) from the error *type* (how a value is incorrect)** and composes
  them into an `ErrorModel`. Requires `pandas>=2.3.0,<2.4.0` — **exactly our
  pandas**, and Python 3.9–3.12. This is the modern replacement for Jenga and
  the right tool for M4/M3's synthetic reference. **Its per-named-column API is
  not documented on the docs landing page — verify in
  `examples/1-Getting-Started.ipynb` before relying on it.** No associated
  paper found: `not found`.

**2024–2026 systems, status.** `TabClean: Reusable LLM-Synthesized Programs for
Tabular Data Cleaning` ([arXiv 2606.25388](https://arxiv.org/html/2606.25388))
states it links code and data on GitHub and is training-free —
**[UNVERIFIED, I did not open the repo]**. `Towards Practical Benchmarking of
Data Cleaning Techniques: On Generating Authentic Errors via Large Language
Models` ([arXiv 2507.10934](https://arxiv.org/abs/2507.10934)) is the closest
thing to a current benchmark-construction standard; code availability
`not found`. GARF (PVLDB 16, [PDF](https://www.vldb.org/pvldb/vol16/p433-peng.pdf))
has no official repo I could find; a third-party experiments repo exists
([S-Eggers/MA-Cleaning-Data-Lakes-in-a-Self-Supervised-Manner](https://github.com/S-Eggers/MA-Cleaning-Data-Lakes-in-a-Self-Supervised-Manner))
and the CDC repo ships a `garf/main.py`.

### 4.2 The minimum credible answer to G3

Running **Raha (#1) + DDC via robpy (#2)** on all three datasets, at a stated
20-label budget, reporting P/R/F1 **and** PR-AUC with our measured error rates
and error types in the caption, closes gap G3 at an effort of roughly one
afternoon. Both are peer-reviewed systems we did not write. Adding the **REIN
harness (#4)** multiplies that by six more detectors for perhaps a day more.
**CDC (#6) is the comparison that matters most scientifically and costs the
most; schedule it separately and do not let it block G3.**

Expected outcome, stated in advance so it cannot be rationalised afterwards:
on **adult**, Raha should score near 0 — its base detectors are the same
rule/outlier family that R1 showed scores F1 0.002 there (HoloDetect's `OD` row),
and SAGED's reproduction has Raha at R 0.10 even on Breast Cancer at 40 % error.
If Raha also scores ~0 on adult, **that is the strongest possible support for our
negative result**: a SIGMOD-award-winning, reproducibility-awarded system with 20
human labels cannot find valid-to-valid label flips either. If it scores well, our
detector is simply behind and we must say so.

---

## 5. Ranked recommendations for W4.5

Five mechanisms, ranked by (expected gain / implementation cost). Each states
**how it could fail** and a **one-number falsification test** that can be run
before any of it is built out.

### M1 (rank 2). Likelihood-ratio rescoring against an explicit corruption model

**Mechanism.** Replace the per-cell score `1 - P(x | rest of row)` with
`log [ P(x | corruption model) / P(x | clean conditional model) ]`. For a
value-swap corruption the natural corruption model is the column's own
**marginal** distribution (a swap draws some other valid value), so the score
becomes `log [ p_marginal(x) / p_conditional(x | row) ]` — *how much worse the
row-conditional model finds this value than the column marginal does*.
Column triage then falls out for free as the mean ratio per column.

**Why.** §3.1. `native_country = Holand-Netherlands` is rare under **both**
models, so the ratio is ~1 and it is correctly not flagged; an `income` flip is
common marginally and improbable conditionally, so the ratio is large. This is
exactly the confound Ren et al. (NeurIPS 2019) identified and fixed, and it is
the formally correct statistic rather than a heuristic.

**Cost. Very low** — one change in the scorer. The marginals already exist in
`DatasetProfile`. **But `DatasetProfile` is built from `clean_df`
(CLAUDE.md §3 soft-leakage caveat), so compute the marginal from dirty data
here**, or the result is not defensible.

**How it could fail.** (a) If the corruption is not marginal-distributed — e.g.
every flip sets `income` to the same value — the assumed corruption model is
wrong and the ratio misranks. (b) On smartfactory the corruption is literally
blank cells, which have ~0 marginal mass, so the ratio is degenerate and the
mechanism contributes nothing there. (c) `p_marginal` and `p_conditional` may
be so correlated that the ranking barely moves — perfectly possible, and
measurable in a single run.

**Falsification test (one number).** Of the top 1,701 adult cells ranked by the
ratio, how many are in `income`? Today it is **0**. Anything > 0 is progress;
R1's derived ceiling for this detector family on adult is precision ≈ 0.2–0.3.
If it is still 0, M1 is dead and we have spent an hour.

### M2 (rank 1 by ratio, run alongside M1). Per-column contamination fraction as a learned column prior

**Mechanism.** For each column *c*, take the existing conditional-surprise
scores for all its cells and estimate the **mixture weight of the anomalous
component**, `pi_hat_c`: either Gao & Tan's exponential+Gaussian mixture
(ICDM 2006; runnable as `pythresh.thresholds.mixmod`) or BBE from Garg et al.
(NeurIPS 2021). Rank columns by `pi_hat_c`; feed `pi_hat_c` as the column prior
in the posterior-odds decomposition of §3.1, and as the prior for Phase 6b's
`q`.

**Why.** Two independent literatures (MPE/PU learning, §2.3; outlier-score
calibration, §2.5) converge on this single quantity, it needs **no labels**, and
it is not confounded by uniform rarity — a uniformly-rare-but-clean column has a
*low* anomalous mixture weight because its rare values are not separable from
its own distribution. It also slots straight into machinery we already have
(`detection_calibrator.py` learns `q` per detection method; this gives it a
per-column prior instead of the default 1.0).

**Cost. Low** — the scores exist; `pythresh` is one `pip install`.

**How it could fail.** (a) **Identifiability at extreme prevalence**: at
smartfactory's 86.6 % corruption the "anomalous" component *is* the majority and
the two-component fit is label-swapped or unidentifiable. (b) BBE's guarantee
holds "whenever we can train a model to cleanly separate out a small subset of
positive examples" — on adult `income` our conditional model tops out near 85 %
accuracy and the corrupted cells are **not** cleanly separable, so the estimator
may collapse toward 0 or toward the model's error rate. (c) `native_country`'s
long rare-value tail may itself fit a two-component mixture and produce a
*false high* `pi_hat`, which would reproduce the current failure exactly.

**Falsification test (one number).** Does `pi_hat_income > pi_hat_native_country`
on adult? If not, the mechanism is dead. (True prevalences for calibration:
`income` 1,701/45,222 ≈ 3.8 % of that column's cells; every other column 0 %.)

### M3 (rank 3). Spend a tiny label budget on COLUMNS, benchmarked against ED2's published Column Selector

**Mechanism.** Treat columns as **strata**. Round-robin a few labels per column
to get `pi_hat_c` with a variance estimate, then allocate the remainder by
**Neyman allocation** (§2.6), or run **nonmyopic active search** with
positives-found as the objective rather than classifier accuracy. Report against
ED2's four published criteria — **Min Certainty, Max Error, Max Prediction
Change, Round-Robin/Random** — which we can reimplement in a few lines each
without ED2's codebase.

**Why.** It is the only recommendation that answers reviewer gap **G3** on the
triage question itself, by comparing against a *published* mechanism instead of
one we wrote. It also corrects W3's objective: at 0.25 % prevalence the task is
active **search**, not active learning.

**Cost. Low-medium** — W3's harness, re-aimed at columns.

**How it could fail — and this one is near-certain on adult.** Expected positives
from 20 uniformly random adult labels = 20 × 0.0025 = **0.05** (*derived*). ED2's
own initialisation requires **≥ 2 known errors and ≥ 2 known correct cells per
column**, which is unreachable for adult's fourteen clean columns, so ED2's
selector would not even start. ED2 also reports instability on its own
lowest-prevalence dataset (Movies, 1 %). **Pre-register the prediction that
label-based column triage finds nothing on adult and only works on breast_cancer
(7.5 %) and smartfactory**; if it does find `income`, that is a genuine surprise
worth a paragraph.

### M4 (rank 4). Column-level two-sample test: observed column vs its model-implied conditional

**Mechanism.** For each column, build a synthetic copy by sampling each cell
from its own conditional predictive distribution, then test observed vs
synthetic: a **classifier two-sample test** (Lopez-Paz & Oquab, ICLR 2017),
or per-column KS/chi-square with **Bonferroni** across the 15 columns (Failing
Loudly, NeurIPS 2019). Equivalently, compute **randomized quantile residuals**
(Dunn & Smyth 1996) per column and test their uniformity. Rank columns by
p-value.

**Why.** It manufactures the missing **reference distribution** by simulation,
which §2's constraint table says is one of only three legal ways to get one; it
yields a calibrated p-value rather than an arbitrary score; and multiple-testing
correction across columns is standard and expected by reviewers.

**Cost. Medium** — needs a sampler from each conditional model plus a test
harness.

**How it could fail.** The null hypothesis is *"the conditional model is
correct"*, so **every misspecified column rejects**, and misspecification is not
corruption. `native_country` is heavy-tailed and will be misspecified under any
model, so it will reject too — reproducing today's failure with a p-value
attached and nothing gained. The only mitigation is Auto-Validate's trick:
calibrate each column's rejection rate on columns *we* corrupted synthetically,
which makes M4 depend on M5.

**Falsification test.** Is adult's `income` p-value smaller than
`native_country`'s? One run of the test on two columns answers it.

### M5 (rank 5). Inject-and-measure: a per-column corruption-response curve

**Mechanism.** For each column *c*, inject *k* % synthetic corruption at several
levels (via `tab-err`, which separates *where* from *how*, or `jenga`), refit,
and record the detector's response Δ_c. Use the *curve* — the mapping from known
injected prevalence to measured response — to read off the observed column's
**implied prevalence**, rather than trusting the brief's raw "responds less when
already polluted" direction.

**Why.** It is the third of the three legal reference sources (§2), and
HoloDetect's **+20 F1 with 3× fewer labels** is the published evidence that
synthetic corruption carries real signal in this exact task.

**Cost. Medium-high** — 15 columns × several injection levels × refit; the most
compute of the five.

**How it could fail.** (a) The "already-polluted columns respond *less*"
direction is an **assumption, not a theorem**: adding fresh corruption to an
already-corrupted column may *increase* the response, because there is now more
signal. **State the predicted sign before running**, and report the sign flip as
a negative result if it comes out the other way. (b) **Circularity** — R1 §R-4's
warning: if the injected corruption resembles the policy that produced
`dirty.csv`, the result is memorisation. Use a different policy and say so.
(c) Compute cost may exceed its value relative to M1/M2, which need no refits.

### Ranking summary

| Rank | Mechanism | Expected gain | Cost | Needs labels? | Dies if |
|---|---|---|---|---|---|
| 1 | **M2** per-column contamination fraction (MPE / mixture weight) | High — supplies the missing prior term and the column ranking at once | Low | No | `pi_hat_income <= pi_hat_native_country` |
| 2 | **M1** likelihood-ratio rescoring | High — fixes the *formally wrong* statistic | Very low | No | top-1,701 still contains 0 `income` cells |
| 3 | **M3** label budget spent on columns, vs ED2's selector | Medium — but it is the one that answers G3 on triage | Low-med | 20 | adult yields ~0.05 expected positives (derived) |
| 4 | **M4** column-level two-sample test | Medium — calibrated p-values, reviewer-friendly | Medium | No | misspecification dominates; `native_country` rejects too |
| 5 | **M5** inject-and-measure response curve | Medium — but it is the only self-contained reference | Med-high | No | response sign is the opposite of predicted; circularity |

**Run M1 and M2 first and together.** They share the existing score matrix,
need no labels and no refits, and each has a single-number falsification test on
a column pair we already understand. If both fail those tests, column triage via
unsupervised scoring is not available on adult, and the honest paper claim
becomes: *per-cell surprise is solved, column localisation of valid-to-valid
corruption is not solvable from the dirty table alone* — which §3.3's negative
result already supports.

### Required amendments to existing docs (do not let these drift)

1. **`docs/literature.md` §1** lists ED2 only as "extracts attribute-, tuple-
   and dataset-level features and uses active learning". It omits the
   **two-stage Column Selector**, which is the single most relevant prior art to
   W4.5. Amend the row and add ED2's four selection criteria.
2. **`docs/research_program.md` §9 / M2** says "the model ranks well within a
   column and cannot tell which column to examine. Per-cell surprise is solved;
   column selection is not." That remains true of *our* system but must no
   longer be written as though the *problem* were unaddressed in the
   literature — ED2 (CIKM 2019) addressed it, and selective editing addressed
   the cost-weighted version in the 1980s–90s.
3. The term to use throughout is **column-level error localisation** (§1.5).

---

## 6. Sources

Read in full or via structured extraction, and quoted above:
- Rabanser, Günnemann, Lipton. **Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift.** NeurIPS 2019. <https://arxiv.org/pdf/1810.11953>
- Neutatz, Mahdavi, Abedjan. **ED2: A Case for Active Learning in Error Detection.** CIKM 2019 · technical report <https://arxiv.org/pdf/1908.06309>
- Ruff et al. **A Unifying Review of Deep and Shallow Anomaly Detection.** *Proc. IEEE* 109(5):756–795, 2021. <https://arxiv.org/abs/2009.11732>
- Ren et al. **Likelihood Ratios for Out-of-Distribution Detection.** NeurIPS 2019. <https://arxiv.org/abs/1906.02845>
- Garg, Wu, Smola, Balakrishnan, Lipton. **Mixture Proportion Estimation and PU Learning: A Modern Approach.** NeurIPS 2021. <https://papers.nips.cc/paper/2021/hash/47b4f1bfdf6d298682e610ad74b37dca-Abstract.html>
- Qiu et al. **Latent Outlier Exposure for Anomaly Detection with Contaminated Data.** ICML 2022, PMLR v162. <https://arxiv.org/pdf/2202.08088>
- Gao, Tan. **Converting Output Scores from Outlier Detection Algorithms into Probability Estimates.** ICDM 2006. <https://cse.buffalo.edu/~jing/doc/ICDM06.pdf>
- Jiang et al. **Efficient Nonmyopic Active Search.** ICML 2017, PMLR 70. <https://icml.cc/virtual/2017/talk/1176>
- Pelleg, Moore. **Active Learning for Anomaly and Rare-Category Detection.** NIPS 2004. <https://papers.neurips.cc/paper_files/paper/2004/file/8c59fd6fbe0e9793ec2b27971221cace-Paper.pdf>
- Song, He. **Auto-Validate: Unsupervised Data Validation Using Data-Domain Patterns Inferred from Data Lakes.** SIGMOD 2021. <https://arxiv.org/abs/2104.04659>
- Wang, Dong, Meliou. **Data X-Ray: A Diagnostic Tool for Data Errors.** SIGMOD 2015. <https://lunadong.com/publication/dataXray_sigmod.pdf>
- Rousseeuw, Van den Bossche. **Detecting Deviating Data Cells.** *Technometrics* 60(2), 2018. <https://arxiv.org/abs/1601.07251>
- Krishnan et al. **ActiveClean: Interactive Data Cleaning For Statistical Modeling.** PVLDB 9, 2016. <https://www.vldb.org/pvldb/vol9/p948-krishnan.pdf>
- Krishnan et al. **BoostClean: Automated Error Detection and Repair for Machine Learning.** <https://arxiv.org/pdf/1711.01299>
- Schelter, Rukat, Bießmann. **JENGA — A Framework to Study the Impact of Data Errors on the Predictions of Machine Learning Models.** EDBT 2021. <https://cdn.amazon.science/ed/04/f752caff482788a214dfb7460c43/jenga-a-framework-to-study-the-impact-of-data-errors-on-the-predictions-of-machine-learning-models.pdf>
- Lopez-Paz, Oquab. **Revisiting Classifier Two-Sample Tests.** ICLR 2017. <https://arxiv.org/abs/1610.06545>
- Goix. **How to Evaluate the Quality of Unsupervised Anomaly Detection Algorithms?** <https://arxiv.org/abs/1607.01152>
- Faltenbacher, Wahl, Herman, Runge. **How PC-based Methods Err.** UAI 2026, PMLR v337. <https://proceedings.mlr.press/v337/faltenbacher26a.html>
- **A Large-scale Study on Unsupervised Outlier Model Selection: Do Internal Strategies Suffice?** <https://arxiv.org/pdf/2104.01422>
- Zhao, Nasrullah, Li. **PyOD: A Python Toolbox for Scalable Outlier Detection.** JMLR 20(96):1–7, 2019. <https://jmlr.org/papers/v20/19-011.html>

Official statistics / error localisation:
- Fellegi, Holt. **A Systematic Approach to Automatic Edit and Imputation.** *JASA* 71(353), 1976 — formulation confirmed via Scholtus, **A generalized Fellegi–Holt paradigm for automatic error localization**, Statistics Canada 12-001-X, 2016. <https://www150.statcan.gc.ca/n1/pub/12-001-x/2016001/article/14538/01-eng.htm>
- de Waal. **Solving the error localization problem by means of vertex generation.** *Survey Methodology* 29(1), 2003. <https://www150.statcan.gc.ca/n1/pub/12-001-x/2003001/article/6609-eng.pdf>
- Statistics Netherlands. **Error localization as a MIP.** <https://www.cbs.nl/-/media/_pdf/2018/30/dp-errorloc-mip-v9.pdf>
- **The Unknown Future of Statistical Data Editing: Some Imputations.** SAGE / *Statistical Journal of the IAOS*, 2025. <https://journals.sagepub.com/doi/10.1177/0282423X251333718>
- **Score Functions for Selective Editing of the U.S. Census Bureau Trade Data.** Census RRS 2012-10. <https://www.census.gov/library/working-papers/2012/adrm/rrs2012-10.html>
- R package **`errorlocate`**. <https://data-cleaning.r-universe.dev/errorlocate/doc/manual.html>

Statistical machinery:
- Dunn, Smyth. **Randomized Quantile Residuals.** *JCGS* 5(3):236–244, 1996. <https://research.usc.edu.au/esploro/outputs/journalArticle/Randomized-quantile-residuals/99449465702621> · R `statmod::qresiduals` <https://rdrr.io/cran/statmod/man/qresiduals.html>
- **Randomized Predictive P-values** (proves the uniformity Dunn & Smyth did not). <https://www.arxiv.org/pdf/1708.08527>
- **A Complexity Measure for Active Learning in Multi-group Mean Estimation.** PMLR v336, 2026. <https://proceedings.mlr.press/v336/aznag26a.html>
- **Near-Exponential Savings for Mean Estimation with Active Learning** (Neyman allocation, `PartiBandits`). <https://arxiv.org/html/2511.05736v1>

Code and tooling:
- [BigDaMa/raha](https://github.com/BigDaMa/raha) · [mohamedyd/rein-benchmark](https://github.com/mohamedyd/rein-benchmark) · [BigDaMa/ExampleDrivenErrorDetection](https://github.com/BigDaMa/ExampleDrivenErrorDetection) · [se-jaeger/conformal-data-cleaning](https://github.com/se-jaeger/conformal-data-cleaning) · [cpitclaudel/dBoost](https://github.com/cpitclaudel/dBoost) · [HoloClean/holoclean](https://github.com/HoloClean/holoclean) · [sis-ethz/Picket](https://github.com/sis-ethz/Picket) · [steverab/failing-loudly](https://github.com/steverab/failing-loudly) · [jiesongk/auto-validate](https://github.com/jiesongk/auto-validate)
- PyPI: `raha` 1.26 · `robpy` 0.0.6 · `tab-err` 0.2.1 · `jenga` · `pyod` · `pythresh` (<https://pythresh.readthedocs.io/en/latest/_modules/pythresh/thresholds/mixmod.html>)
- `robpy.outliers.DDC` docs: <https://robpy.readthedocs.io/en/stable/api/robpy.outliers.html>

### Flagged as unverified or not found

- **Uni-Detect's mechanism** (null hypothesis, statistic, corpus calibration,
  any P/R numbers): the PDF body is FlateDecode-compressed and could not be
  extracted. The related patent describes a row-subset-removal perturbation test
  but is a different document. **[UNVERIFIED]**
- **LOE's robustness to a 5-percentage-point error in the assumed anomaly ratio**:
  reported by a citing paper, not by the LOE paper. **[UNVERIFIED]**
- **Whether `jenga`'s and `tab-err`'s corruptions can target a named column**:
  not documented on the pages I read. Must be checked in the code/notebooks.
- **Selective editing's efficiency figures** (5 % → 90 %, top 20 % → 90 %,
  ≥ 50 % cost savings, ≤ 20 % of production resources): all quoted from the 2025
  SAGE overview, which attributes them to underlying studies I did not read.
  **Cite via the overview, not as primary.**
- **TabClean's released code**: claimed in the paper, repo not opened.
  **[UNVERIFIED]**
- **SAGED, ZeroED, ForestED code**: `not found`.
- **ED2 per-dataset label counts**: in figures only, not quantified in the text.
  `not found` as numbers.
- **`pythresh` publication**: no peer-reviewed paper found; cite the docs.
  Author/maintainer is KulikDM, **not** PyOD's Zhao. `not found`.
- **Data X-Ray scoring**: Bayesian cost model, **not** MDL. My prior expectation
  of an MDL formulation was wrong and is corrected in §1.3.
- **BoostClean's selector**: boosting, **not** a multi-armed bandit. Corrected
  in §2.7.
