# CLAUDE.md — Risk-Aware LLM-Assisted Data Repair

Project handoff context. Read this before touching any code.

---

## 1. What this project is

An end-to-end pipeline that finds suspicious cells in a tabular dataset, proposes
repair candidates, quantifies how uncertain those repairs are, **calibrates** that
uncertainty into a trustworthy probability, and then makes a **cost-sensitive
decision**: repair automatically, route to a human, or leave the cell alone.

**The research claim is not "we repair data well."** It is:

> A calibrated probability of correctness, combined with explicit costs of being
> wrong, produces better decisions than acting on raw model confidence. The same
> confidence should yield different actions when the consequences change.

Everything in the codebase serves that claim. Changes that improve repair accuracy
but break the calibration→cost→decision chain are not improvements.

Academic context: VIT Vellore final-year project (Second Review completed).
Intended tangible outcome is a conference paper.

---

## 2. Pipeline architecture

```
dirty.csv
   │
   ├─ Phase 1-2  src/data/loader.py, profiler.py, schema.py
   │             → DatasetProfile (column kinds, quartiles, vocabularies)
   │
   ├─ Phase 3    src/detection/detector.py
   │             → DetectionRecord per suspicious cell
   │             Six checks: missing-token, out-of-range, IQR fence (k=3),
   │             format, novel-category, whitespace
   │
   ├─ Phase 4    src/repair/mock_repair.py
   │             → RepairResult with top-k RepairCandidate (value, probability)
   │             CURRENTLY A MOCK: difflib similarity + frequency weighting +
   │             median/quartile imputation + seeded noise. NOT an LLM.
   │
   ├─ Phase 5    src/uncertainty/estimator.py
   │             → UncertaintyRecord: raw_confidence, self_consistency,
   │               entropy, normalised_entropy
   │
   ├─ Phase 6    src/calibration/calibrator.py
   │             → calibrated probability p via Platt/logistic or isotonic
   │             THE ONLY PLACE clean_df LEGITIMATELY ENTERS (0/1 labels only)
   │
   ├─ Phase 6b   src/calibration/detection_calibrator.py
   │             → q = P(flagged cell is genuinely corrupted)
   │             Learned per detection_method from 0/1 corruption labels.
   │             SECOND legitimate clean_df entry point (labels only).
   │
   ├─ Phase 7    src/risk/cost_model.py
   │             → RiskCalculator, ExpectedCosts, RiskRecord
   │             Actions: ACCEPT_REPAIR / HUMAN_REVIEW / LEAVE_UNCHANGED
   │
   ├─ Phase 8    src/decision/policy.py
   │             → DecisionPolicy, DecisionResult
   │             Actions: AUTO_REPAIR / HUMAN_REVIEW / ABSTAIN
   │             Bridges to Phase 7 via DecisionPolicy.from_cost_config()
   │
   ├─ Phase 9    src/baselines/baselines.py
   │             → AlwaysAutoBaseline, ConfidenceThresholdBaseline,
   │               sweep_thresholds(), wrap_proposed_method()
   │
   └─ Phase 10   src/evaluation/{repair,calibration,cost}_metrics.py
                 → RepairEvaluator, CalibrationEvaluator, CostEvaluator,
                   compare_policies()
```

### Decision formulas

For calibrated probability of correctness `p` and calibrated probability that the
flagged cell is genuinely corrupted `q`:

```
E[AUTO_REPAIR]  = (1 - p) * C_wrong
E[HUMAN_REVIEW] = C_human           (constant in p and q)
E[ABSTAIN]      = q * C_abstain     (constant in p, scales with q)

action = argmin over the three
```

Tie-break priority (documented in `policy.py`): `AUTO_REPAIR > HUMAN_REVIEW > ABSTAIN`.

`q` defaults to 1.0 everywhere, which reproduces the original constant formula
bit-for-bit. It is learned by `src/calibration/detection_calibrator.py` (Phase 6b).
Derived decision boundaries, verified against the implementation on a 201x201 grid
with zero mismatches:

```
AUTO vs HUMAN      p* = 1 - C_human / C_wrong        (vertical)
HUMAN vs ABSTAIN   q* = C_human / C_abstain          (horizontal)
AUTO vs ABSTAIN    p  = 1 - q * C_abstain / C_wrong  (sloped)
```

**Honest limitation, state it in the paper:** `q*` is horizontal, so for a single
cell with fixed `q` one of HUMAN/ABSTAIN is still unreachable — both are constant
in `p` and never cross. What the mechanism changes is *which* of the two is live,
per cell, on evidence. "All three actions reachable" is a claim about the
population, not about every cell. See `docs/cost_semantics.md`.

Costs live in `configs/costs.yaml`. They are **experimental assumptions in abstract
penalty units**. Never describe them as monetary, clinical, or as supplied by the
REIN benchmark — the codebase is careful about this and papers/docs must stay careful.

---

## 3. Non-negotiable invariant: no ground-truth leakage

`clean.csv` is ground truth. It may be used **only after** predictions exist.

**Forbidden:**

- Passing `clean_df` into any detection, repair, uncertainty, decision or baseline function
- Comparing a candidate against the clean value during generation or ranking
- Sending clean values to a real LLM (Phase 12)
- Hard-coding known-correct answers

**Permitted:**

- `CalibrationBuilder.build()` reads `clean_df` to produce 0/1 *repair-correctness*
  labels (feeds `p`)
- `DetectionLabelBuilder.build()` reads `clean_df` to produce 0/1 *cell-corrupted*
  labels (feeds `q`) — same permitted use, symmetric with the above
- All of `src/evaluation/` reads `clean_df` to score decisions after the fact

Both calibrators are fitted on a train split and predict for all records; neither
`fit` nor `predict` ever receives clean data. `decide_all()` has no ground-truth
parameter by construction.

There are existing tests asserting no-leakage. Keep them passing and add equivalents
for any new module.

**Known soft-leakage caveat (document, don't silently "fix"):** `DatasetProfile` is
built from `clean_df`, so detector IQR fences / valid ranges / vocabularies, and the
repairer's imputation medians and quartiles, are derived from clean aggregate
statistics. The code documents this as acceptable population-level schema knowledge.
A reviewer will challenge it. Preferred resolution is an ablation computing those
stats from dirty data and reporting the gap — not a quiet code change that
invalidates already-reported numbers.

---

## 4. Current state

**Phases 1–11 complete, plus Phase 6b (calibrated abstention). Phases 12–13
not started.**

Repo: `github.com/sruthi-pyda/data-repair-project`. Three stacked branches off
`main` (HEAD `5d8cc22`), none merged yet:

```
main
 └─ fix/known-issues-phase5      §5.1–5.4 fixes
     └─ feat/phase11-dashboard   Phase 11
         └─ feat/calibrated-abstention   Phase 6b  ← current tip
```

Tests: **1268 passing** on pandas 2.3 (~3 min).

| File | Tests | Phase |
|---|---|---|
| `test_loader.py` | 33 | 1 |
| `test_profiler.py` | 66 | 2 |
| `test_detector.py` | 54 | 3 |
| `test_mock_repair.py` | 85 | 4 |
| `test_uncertainty.py` | 74 | 5 |
| `test_calibration.py` | 80 | 6 |
| `test_cost_model.py` | 102 | 7 |
| `test_policy.py` | 103 | 8 |
| `test_baselines.py` | 87 | 9 |
| `test_evaluation.py` | 133 | 10 |
| `test_pipeline_runner.py` | 38 | 11 |
| `test_plots.py` | 58 | 11 |
| `test_app.py` | 30 | 11 |
| `test_detection_calibrator.py` | 76 | 6b |
| `test_calibrated_abstention.py` | 104 | 6b |
| `test_abstain_realised_cost.py` | ~30 | 6b |
| `test_decision_regions.py` | 49 | 6b |
| `test_three_actions_e2e.py` | 12 | 6b |

(Phase 2/6/7/8 files also gained regression tests for the §5 fixes.)

**Phase 11 / 6b modules:**

```
src/pipeline/runner.py      ← run_pipeline / decide_all / evaluate_policies
                               shared by app.py and (future) run_experiment.py
src/visualization/plots.py  ← figure helpers incl. decision_map (p,q)
src/calibration/detection_calibrator.py  ← learns q (Phase 6b)
app.py                      ← Streamlit dashboard: `streamlit run app.py`
experiments/three_actions_ablation.py    ← main results table
experiments/smoothing_ablation.py        ← prior-robustness table
results/tables/three_actions_ablation.md ← generated output
docs/cost_semantics.md      ← what each cost means; READ BEFORE writing results
docs/three_actions_plan.md  ← the Phase 6b design record
.claude/launch.json         ← dev-server config for the desktop-app browser
```

**Empty stub files (Phases 12–13 not begun):**

```
run_experiment.py           ← end-to-end runner; build on src/pipeline/runner.py
src/repair/llm_repair.py    ← real LLM integration
src/uncertainty/uncertainty.py   ← dead stub; estimator.py is the real module
```

Note `src/repair/prompts.py` (335 lines) already exists — Phase 12 prompt
scaffolding is written, the API client is not.

### Verified results (reproduced, mock mode)

Calibration:

| Dataset | Method | Brier | ECE |
|---|---|---|---|
| breast_cancer | raw | 0.2033 | 0.1171 |
| breast_cancer | isotonic | 0.1883 | 0.0530 |
| smartfactory | raw | 0.2780 | 0.3250 |
| smartfactory | logistic | 0.1710 | 0.0158 |
| adult | — | degenerate | degenerate |

Breast Cancer empirical cost (355 records, wrong=50/review=5/leave=10):

```
Policy (risk model)   1620.00   mean 4.5634
Always ACCEPT         6000.00   mean 16.9014
Always REVIEW         1775.00   mean 5.0000
Always LEAVE          3550.00   mean 10.0000
Top-1 correct: 235 (66.2%)
```

**Headline ablation result** (`results/tables/three_actions_ablation.md`,
realised cost in experimental units, seed 42, full datasets):

| scenario | original | proposed | always-auto | best threshold |
|---|---|---|---|---|
| breast_cancer k=3 | 1615 | **1615** (unchanged by design) | 5800 | 1720 |
| breast_cancer k=1.5 | 2675 | **1750** | 15450 | 2695 |
| smartfactory outliers off | 14186 | **14186** (unchanged) | 126275 | 14186 |
| smartfactory outliers on | 85464 | **14186** | 1017250 | 85464 |
| adult | 11658 | **0** ‡ | 58290 | 11658 |

‡ zero *residual-corruption* cost on the flagged cells only; Adult's 1,701
genuine corruptions were never detected. Always report both.

The learned `q` accounts for 100% of every reduction. The evaluator change
alone contributes 0 and is a precondition, not an independent gain. Raw
`detection_score` as `q` reproduces almost nothing. Smoothing changes nothing.
The one place the method does harm: at k=1.5, 5 of 195 abstentions land on
genuinely corrupted cells.

**Adult is a deliberate negative result, not a bug.** Its true corruption is
income-label flips (valid value → valid value), undetectable from dirty data alone.
Phase 3 detects only `fnlwgt` IQR false positives, so all correctness labels are 0,
calibration has no signal, and both calibrators correctly refuse to fit. This is a
genuine finding — a responsible pipeline must support "not detected" rather than
repairing every statistical outlier. Preserve it; do not tune it away.

---

## 5. Known issues — fix these first

**Status: 5.1, 5.2, 5.3 and the first bullet of 5.4 are all resolved.** The text
below is kept as the record of what was wrong and why each fix was chosen. Still
open: repeated-split confidence intervals and the 500-record demo cap (both
Phase 13).

5.2 was resolved properly, not by the cheap config tweak: see
`docs/cost_semantics.md` and `docs/three_actions_plan.md`. Adult's
`c_leave_error` is back to the original 5; abstention comes from the learned `q`.
The detector-confidence version suggested below was **tested and rejected** — it
changes zero decisions, because `detection_score` is a fixed per-check-type
constant and not a probability (Adult outliers score ~0.6 while 0% of them are
genuinely corrupted). That negative result is variant V4 of the ablation.

Multivariate calibration result (mock mode, single split, seed 42):
breast_cancer Brier 0.2039→0.1983 / ECE 0.1351→0.0650 vs single-feature
logistic (isotonic still best on Brier); smartfactory 0.1684, best method.
`self_consistency` gets ~0 weight on both — in mock mode all the gain comes
from entropy. Report this; do not hide it.

### 5.1 Two failing tests are a pandas 3.x incompatibility

```
tests/test_profiler.py::TestColumnKind::test_bc_bare_nuclei_is_mixed
tests/test_profiler.py::TestCorruptionCharacterisation::test_bc_bare_nuclei_has_format_errors
```

Root cause, confirmed: `_classify_column_kind()` in `src/data/profiler.py` gates the
MIXED check on `series.dtype == object`. Under pandas 3.0 with
`future.infer_string = True`, a text column's dtype is `str`, not `object`, so the
branch is skipped and `Bare Nuclei` falls through to CATEGORICAL — even though 97.7%
of its values parse as numbers, well above the 0.8 MIXED threshold.

Consequence: format errors (`1a0`, `a10`, `??`) in the flagship Breast Cancer demo
column are never characterised.

Fix: replace the `== object` check with a dtype-agnostic test
(`pd.api.types.is_string_dtype(series)` or `not is_numeric_dtype(series)`), and pin
pandas in `requirements.txt` (currently unpinned `pandas>=2.0.0`). Audit for other
`== object` comparisons while you are in there.

This passes on pandas 2.x, which is why the README claims 494 passing.

### 5.2 ABSTAIN is unreachable under every shipped cost config

`E[HUMAN_REVIEW]` and `E[ABSTAIN]` are both constant in `p`, so whichever is smaller
wins at every confidence. All three dataset configs set
`c_leave_error > c_human_review`, so ABSTAIN can never be selected.

Swept `p` from 0.000 to 1.000 in 0.001 steps:

```
adult          wrong=10 human=2 abstain=5   → HUMAN_REVIEW 800, AUTO_REPAIR 201, ABSTAIN 0
breast_cancer  wrong=50 human=5 abstain=10  → HUMAN_REVIEW 900, AUTO_REPAIR 101, ABSTAIN 0
smartfactory   wrong=25 human=2 abstain=5   → HUMAN_REVIEW 920, AUTO_REPAIR  81, ABSTAIN 0
```

**The policy code is correct** — `test_policy.py` proves ABSTAIN fires when
`c_abstain < c_human`. This is purely a `configs/costs.yaml` problem, and Phase 10
comparison tables will print 0% abstention on every row.

Cheapest defensible fix: lower `c_leave_error` for Adult (e.g. `1`), justified by the
fact that Adult detections are false positives — the cell is not actually corrupted,
so leaving it alone costs almost nothing. Add the justification to the YAML `note`.

A stronger, paper-grade fix is to make abstention `p`-dependent, e.g.
`E[abstain] = P(cell is genuinely corrupted) * C_leave_error`, using detector
confidence. This makes all three arms non-degenerate. Discuss before implementing —
it changes every reported cost number.

### 5.3 Uncertainty signals are computed but unused

`CalibrationSample` carries only `raw_confidence`. Self-consistency and entropy are
computed in Phase 5 and then discarded. Feeding all three into a multivariate
calibrator is a cheap, real contribution and a natural ablation table.

### 5.4 Minor

- Adult's logistic row prints `Brier 0.0000 / ECE 0.0000` while flagged not-fitted.
  In a results table this reads as a perfect score. Print `—` instead.
- Calibration uses one 70/30 split. Breast Cancer has ~392 samples. Report repeated
  splits with confidence intervals, not a single number.
- Demo scripts cap Adult and Smart Factory at 500 records. Acceptable for a demo,
  not for final paper results.

---

## 6. Remaining work

### Phase 11 — Streamlit dashboard (`app.py`, `src/visualization/plots.py`)

**Done** on `feat/phase11-dashboard`. `streamlit run app.py`, offline, no key.
The original spec follows; all of it is implemented. Verified by
`tests/test_app.py` (AppTest, headless) including the slider-flips-decision
interaction. (The Windows cp1252 crash in the demo scripts is fixed: every script
calls `src/console.py::enable_unicode_stdout()`; no env var needed.)

Sidebar: dataset selector, mock/real mode, three cost sliders, confidence threshold
for the baseline, n samples, random seed.

Sections: dataset overview → one dirty record → candidate table → uncertainty panel →
expected costs (label them "experimental cost units") → prominent final decision →
**interactive risk demonstration** → baseline comparison → plots.

Section 7 is the centrepiece: moving the cost sliders must recompute expected costs
and flip the displayed action live. That single interaction is the whole thesis.

Leakage rule applies to the UI: never display the clean value beside a dirty cell in
the decision demo. Ground truth only in a clearly separated evaluation section.

### Phase 6b — Calibrated abstention (done, branch `feat/calibrated-abstention`)

`E[ABSTAIN] = q * C_abstain` with `q` learned per detector check type. Rationale,
cost definitions, derived regions and the smoothing ablation are in
`docs/cost_semantics.md`; the step-by-step record is `docs/three_actions_plan.md`.

Key facts for the paper:

- `C_abstain` (config key `c_leave_error`) is **corruption-persistence cost**:
  charged only when a genuinely corrupted cell is left unrepaired. This is the
  project's own original definition; the pre-branch evaluator charged it
  unconditionally, which was inconsistent with it.
- Any per-flag operational cost is decision-invariant (adds a constant to all
  three arms) and must be reported separately as `n * C_flag`, never folded into
  `C_abstain`.
- Adult's realised cost of 0 means **zero residual-corruption cost on the flagged
  cells only**. Adult has 1,701 genuine corruptions that the detector never
  found; they enter no arm of the objective. Always report that alongside.
- Calibration failure defaults to `q = 1`, preserving the pre-calibration policy.
  It is a conservative assumption, not evidence.
- Genuine corruptions per dataset (for results tables): breast_cancer 630,
  smartfactory 7,093, adult 1,701.

### Phase 12 — Real LLM (`src/repair/llm_repair.py`)

`prompts.py` already exists. Needs the API client, structured-output parsing, and
validation of candidate structure / probability ranges / normalisation / duplicates /
empty lists. API key from env var via python-dotenv; update `.env.example`. Must fall
back to mock mode cleanly and report the fallback on: no key, request failure,
malformed response, invalid probabilities.

Tests must mock the API. No test may depend on a live call.

Send only: dataset/domain, column name, dtype, dirty value, nearby row context,
representative valid values, schema constraints. Never the full dataset. Never the
clean value.

**This phase is optional for the review but mandatory for the paper** — the title
says "LLM-Assisted" and there is currently not one real model call in the project.

### Phase 13 — Experiments, results, docs

`run_experiment.py` is empty, so `python run_experiment.py --dataset adult --mode mock`
does not work. Build it first — every other phase's reproducibility depends on it.

Produce `results/{figures,tables,predictions}/` with fixed seeds. Four table groups:
repair quality, calibration (raw vs calibrated), risk-aware performance, policy
comparison. Plots: reliability diagrams, confidence distributions, cost comparison,
action distributions, baseline vs proposed.

Do not fabricate results. If the proposed method loses on a dataset, report it.

---

## 7. Commands

```bash
# install
pip install -r requirements.txt

# full test suite
python -m pytest -q

# single phase
python -m pytest tests/test_policy.py -v

# working demos (Phases 4-7)
python demo_repair.py
python demo_uncertainty.py
python demo_calibration.py
python demo_risk.py

# Phase 11 dashboard (offline, mock mode)
streamlit run app.py

# Experiments (deterministic, seed 42)
python experiments/smoothing_ablation.py
python experiments/three_actions_ablation.py

# NOT YET WORKING — Phase 13 must build this (use src/pipeline/runner.py)
python run_experiment.py --dataset adult --mode mock
```

Datasets live in `datasets/{adult,breast_cancer,smartfactory}/{clean,dirty}.csv`.
Config in `configs/default.yaml` and `configs/costs.yaml`. Random seed 42 throughout.

---

## 8. Conventions

- Typed dataclasses for every record type; `to_dict()` for serialisation
- Module-level docstrings explaining purpose, leakage rules, and worked examples —
  match this style, it is the codebase's strongest asset
- Deterministic seeding derived from `(dataset, column, row_id)`
- Tests are extensive and behavioural. Add tests in the same style for new modules,
  including edge cases (empty input, p=0, p=1, invalid probabilities, ties) and an
  explicit no-leakage assertion
- Do not rewrite completed phases. Phase 8 correctly wraps Phase 7 rather than
  duplicating it — follow that pattern
- Never claim a phase is complete without actually running the tests and the demo

---

## 9. Collaboration

Two-person project. Sruthi Pyda wrote Phases 1–10. **As of 2026-09-22 the project
is under Saanvi's control**; Sruthi's sign-off is no longer required before
changing shared interfaces (`RiskRecord`, `UncertaintyRecord`, `CostConfig`,
`DecisionResult`) or the cost configs.

The replacement rule is evidence, not permission: **a change is kept only if it is
objectively better for the paper's claim, measured.** If it is not measurably
better, revert it. Every change on `feat/calibrated-abstention` is backed by a
before/after measurement; the detector-confidence abstention idea was rejected
under exactly this rule.

Work on a branch, not directly on `main`.
