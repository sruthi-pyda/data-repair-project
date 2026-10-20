# Wave 8 plan — a novel mechanism for column triage, from the literature's own factors

Committed before any W8 code. Three mechanisms, tested in parallel on disjoint
files, each with falsification fixed here.

---

## 1. What we are fixing, precisely

W7.1e settled the diagnosis:

| | |
|---|---|
| **within** the corrupted column | we capture **77.9 %** of the Bayes-optimal signal (0.5626 of 0.7222 on adult `income`) |
| **across** columns | PR-AUC **0.0025** over all 678,330 cells — chance |

So per-cell scoring inside a column is nearly exhausted (~1.28x headroom) and
**the entire remaining loss is choosing the column.** Three mechanisms have
died trying: M1 likelihood ratio, self-supervised injection (W5.3,
cardinality confound), M2 contamination fraction (W6.1, lost to the trivial
baseline).

And the hard instance is sharper than "rarity": adult's corruption is
**entirely one-directional**, all 1,701 flips `1 -> 0`, majority class to
minority. A corrupted cell is marginally indistinguishable from an ordinary
minority-class cell; among cells showing `0`, only ~13 % are corrupted.

---

## 2. Decomposition — where the published edge actually comes from

From `docs/literature.md` plus this wave's search. Sources at §7.

| factor | evidence | do we have it? |
|---|---|---|
| **F1. Row-context conditioning** — `P(cell \| rest of row)` not `P(cell)` | Wave 1; AimNet; Picket | **yes** — and it is why we reach 78 % of ceiling within a column |
| **F2. Learn the corruption model from the dirty data** | HoloDetect: learned augmentation **0.953 F1** on Soccer vs **0.166** for random transformations at 10 % training data | **no** — our generator applies *designed* types, not learned policies |
| **F3. Many weak detectors as a feature VECTOR, then classify** | Raha's core architecture; ED2 | **no** — Wave 2 did noisy-OR fusion, which is far weaker than features-into-a-classifier |
| **F4. Three representation scopes** — attribute, tuple, dataset | HoloDetect, ED2 | **no** — we use one scalar per cell |
| **F5. A few labels placed well** | Raha: 2 labels ≈ 0.40 F1, 20 ≈ 0.85 | tested, **fails at 0.25 % prevalence** (Wave 2/3) |
| **F6. Masked-column self-supervision** | arXiv 2309.08374: masked-column prediction beats contrastive on tabular; **no** SSL method beats the baseline on *global* anomalies | partially (our conditional model is this) |
| **F7. Calibration of the detector score** | HoloDetect uses Platt | **yes**, and *nobody evaluates calibration quality* — our niche |

**The single biggest unexploited factor is F2**, and HoloDetect's ablation is
the warning attached to it: **random transformations do not work** (0.166 vs
0.953). A corruption policy must be *learned from the dirty table*, not
designed. That is a direct risk to our generator-based plans.

---

## 3. The fundamental observation this wave rests on

Write the dirty table's generative process per column `c`. Corruption is
applied to randomly chosen rows, **independently of the row's other values**;
the truth is not. So the observed value distribution is a two-component
mixture with an **asymmetry that identifies it**:

    P_obs(v | x)  =  (1 - r_c) * f(v | x)      <- clean: DEPENDS on x
                   +      r_c  * g(v)          <- corruption: does NOT depend on x

- `f(v|x)` — the clean conditional (what our Wave 1 model already estimates)
- `g(v)` — the corruption's value distribution
- `r_c` — the column's contamination rate

**x-independence is the identifying restriction.** It is the one structural
fact that distinguishes corruption from legitimate rarity, and none of our
four failed mechanisms used it.

The posterior then falls out, and it is a **probability, not a score**:

    P(corrupt | v, x)  =         r_c * g(v)
                          -----------------------------------
                          (1-r_c) * f(v|x)  +  r_c * g(v)

Why this is the right shape for both of our failures:

- **rare-but-valid** — a rare correct country has *small* `g(v)` (corruption
  does not concentrate there) and *relatively large* `f(v|x)` (the row implies
  it) → posterior **low**. Correct.
- **the real adult case** — a flipped `income` has *large* `g(v)` (all
  corruption mass sits on `0`) and *small* `f(v|x)` (the row implies `1`) →
  posterior **high**. Correct.
- **cross-column comparability** — it is a posterior probability by
  construction, so it needs no per-column normalisation. That is exactly what
  every failed mechanism was trying to manufacture.

It also explains **why M2 failed**: M2 fitted a mixture to the *scores*, a
1-D summary that has already destroyed the x-dependence/x-independence
asymmetry. The mixture has to be fitted in `(v given x)` space, which is where
the identifying restriction lives.

And `g` being *estimated* rather than assumed is precisely F2 — a corruption
policy learned from the dirty data, which HoloDetect's ablation says is the
difference between 0.953 and 0.166.

### Prior art — state it, do not overclaim

**Kubica & Moore, *Probabilistic Noise Identification and Data Cleaning*
(CMU, 2002/2003)** is the closest: a clean model, a noise model and a
corruption matrix fitted by EM-style alternating optimisation. It is the
honest citation and must appear wherever this mechanism is described. What is
different here: a **per-column conditional-plus-marginal** mixture with
**x-independence as the explicit identifying restriction**, yielding a
**cross-column-comparable calibrated posterior** that feeds a cost-sensitive
three-way decision. Also related: noisy-label identifiability via multinomial
mixtures (arXiv 2301.01405).

The claim is therefore **not** "a new way to model noise". It is "x-independence
identifies the contamination, which makes per-cell error probabilities
comparable across columns, which is what column triage needs".

---

## 4. The three mechanisms (parallel, disjoint files)

### W8-A — CACM, contamination-aware conditional mixture (§3)

`src/detectors/cacm.py`. Per column, EM over the mixture above:

- E-step: posterior responsibility that each cell is corrupt
- M-step: refit `f` on responsibility-weighted clean cells, refit `g` as the
  responsibility-weighted value histogram, update `r_c`

Initialise `r_c` small (e.g. 0.05) and `g` as the column marginal. Highest
novelty; directly targets both failures.

### W8-B — learned-policy synthetic supervision (F2)

`src/detectors/synth_supervised.py`. Train a supervised per-cell classifier on
corruption whose **policy is estimated from the dirty table** rather than
designed — which transition types actually occur, at what rate, into which
values. Features must be cross-column comparable (ranks/quantiles of the
conditional score, not raw scores). The pre-registered risk is HoloDetect's
0.166: if the policy is effectively random, this fails.

### W8-C — Raha-style feature-vector ensemble (F3 + F4)

`src/detectors/feature_ensemble.py`. Build a per-cell **feature vector** from
every detector we already have (rule checks, conditional, marginal, ratio,
per-column rank, cardinality, tuple-level aggregates), then classify using
synthetic labels. This is the published winner's architecture; Wave 2 only
ever did noisy-OR over two detectors.

---

## 5. Falsification — fixed now, for all three

A mechanism is kept only if **all** of these hold. Anything less is reported
as a failure.

1. **Column triage.** Pooled column-level AUC must beat the mean-raw-score
   baseline, recomputed on the same scored columns as the mechanism.
   W6.1's baseline was **0.7512**; it must be **recomputed**, not carried over,
   because eligibility will differ (the W6 lesson).
2. **Paired, not averaged.** Compare per dataset with a sign/Wilcoxon test
   across datasets. A mean of AUCs weights a 2-column table equally with a
   19-column one and carries no interval.
3. **Cell level on the real hard case.** Must beat our current adult
   whole-table PR-AUC (**0.0025**) and be reported against the real-corruption
   within-column ceiling (**0.7222**, `results/w7_real_ceiling.md`).
4. **Not just rarity.** Report the correlation between the statistic and each
   column's rare-value fraction. If rarity explains the ranking, it is the
   original failure again.
5. **Not just coverage.** Detector-skipped columns excluded, and the baseline
   restricted to the same columns (the W6.1 pre-fix bug).
6. **Not just cardinality.** Report the correlation with column cardinality
   (the W5.3 confound).
7. **No leakage.** Fitted on the dirty table only, `profile=None`. Clean data
   for scoring afterwards, or for a clearly-labelled oracle.

---

## 6. Tasks

- [ ] Commit this plan before any W8 code
- [ ] **W8-A** CACM — `src/detectors/cacm.py` + tests
- [ ] **W8-B** learned-policy synthetic supervision + tests
- [ ] **W8-C** feature-vector ensemble + tests
- [ ] **W8-D** one shared harness evaluating all three against §5, so the
      comparison is on identical units
- [ ] Record the verdict per mechanism, pass or fail

---

## 7. Sources

- HoloDetect, SIGMOD 2019 — [arXiv 1904.02285](https://arxiv.org/pdf/1904.02285).
  Learned augmentation 0.953 F1 vs random transformations 0.166 on Soccer at
  10 % training data (Table 4); "+20 F1" is an average across methods, not an
  ablation of augmentation alone.
- Kubica & Moore, *Probabilistic Noise Identification and Data Cleaning*, CMU
  [2002](https://publications.ri.cmu.edu/storage/publications/pub_files/pub3/kubica_jeremy_martin_2002_1/kubica_jeremy_martin_2002_1.pdf),
  [2003](https://publications.ri.cmu.edu/storage/publications/pub_files/pub4/kubica_jeremy_martin_2003_4/kubica_jeremy_martin_2003_4.pdf)
  — closest prior art for §3.
- *Understanding the limitations of self-supervised learning for tabular
  anomaly detection* — [arXiv 2309.08374](https://arxiv.org/pdf/2309.08374).
  Masked-column prediction beats contrastive; no SSL method beats the baseline
  on global anomalies.
- SemanticMask, IJCAI 2024 — [PDF](https://www.ijcai.org/proceedings/2024/0262.pdf).
- *Towards the Identifiability in Noisy Label Learning* —
  [arXiv 2301.01405](https://arxiv.org/pdf/2301.01405).
- Conformal Data Cleaning, AISTATS 2024 —
  [PDF](https://proceedings.mlr.press/v238/jager24a/jager24a.pdf). Per-column
  conditional models with calibrated uncertainty; the standing novelty risk.
- Note from this wave's search: **no published cell-level error-detection
  benchmark reports PR-AUC.** Our PR-AUC-primary rule is right for extreme
  imbalance but makes us incomparable, so report precision/recall/F1 too.
