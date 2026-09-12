# Risk-Aware Human-in-the-Loop Framework for Trustworthy LLM-Assisted Data Repair

A calibrated, cost-sensitive pipeline that detects data errors, proposes repairs using a mock LLM, estimates uncertainty, calibrates confidence, and recommends the minimum-expected-cost action — **ACCEPT**, **HUMAN REVIEW**, or **LEAVE UNCHANGED** — for every dirty cell.

---

## Table of Contents

1. [Project Objective](#1-project-objective)
2. [Research Question](#2-research-question)
3. [Datasets](#3-datasets)
4. [Problem Being Addressed](#4-problem-being-addressed)
5. [Methodology and Pipeline](#5-methodology-and-pipeline)
6. [Project Phases](#6-project-phases)
7. [Technologies Used](#7-technologies-used)
8. [Project Structure](#8-project-structure)
9. [Installation](#9-installation)
10. [How to Run](#10-how-to-run)
11. [Running the Test Suite](#11-running-the-test-suite)
12. [Expected Outputs](#12-expected-outputs)
13. [Configuration](#13-configuration)
14. [Key Design Decisions and Limitations](#14-key-design-decisions-and-limitations)
15. [Conceptual Q&A](#15-conceptual-qa)

---

## 1. Project Objective

Large Language Models can generate plausible data repairs but can also be confidently wrong. Blindly applying every suggestion is risky; always routing every cell to a human reviewer is expensive.

This project builds a **calibrated, cost-sensitive decision layer** that, for each detected dirty cell, chooses the action with the **lowest expected penalty cost**:

| Action | Triggered when |
|---|---|
| `ACCEPT_REPAIR` | Calibrated probability of correctness is high enough that auto-repair is cheaper than review |
| `HUMAN_REVIEW` | Cost of a wrong repair outweighs the human-review cost |
| `LEAVE_UNCHANGED` | When leaving the error unrepaired is cheaper than both alternatives |

The goal is to **minimise total expected repair cost**, not merely maximise repair accuracy.

---

## 2. Research Question

> Can a calibrated, cost-sensitive policy reduce total expected repair cost compared to a naive "always auto-repair" baseline and a fixed confidence-threshold baseline?

---

## 3. Datasets

Three datasets from the [REIN benchmark](https://github.com/rein-benchmark) are used, each representing a different domain and corruption profile:

| Dataset | Domain | Rows | Columns | Corruption type |
|---|---|---|---|---|
| `adult` | Socioeconomic | 45,222 | 15 | Label noise — `income` column flipped (1 ↔ 0) |
| `breast_cancer` | Healthcare (UCI) | 699 | 12 | Missing values, out-of-range numerics, format errors |
| `smartfactory` | Industrial IoT | 23,645 | 19 | Sensor dropout — missing values only |

**Folder layout:**
```
datasets/
  adult/          clean.csv  dirty.csv
  breast_cancer/  clean.csv  dirty.csv
  smartfactory/   clean.csv  dirty.csv
```

**Ground-truth rule:** `clean.csv` is accessed **only inside evaluation modules**. It is never passed to error detection, repair, uncertainty estimation, or the risk model. This enforces a strict no-data-leakage boundary.

---

## 4. Problem Being Addressed

Real-world tabular datasets suffer from corruption: missing values, out-of-range numerics, malformed categories, and label noise. Automated repair systems face three failure modes:

1. **Confidently wrong repairs** — the model says 90% confidence but is incorrect.
2. **Uniform routing** — always auto-repair wastes human trust; always routing to humans is too expensive.
3. **Uncalibrated confidence** — raw model scores do not reflect true correctness probabilities.

This project addresses all three by combining **uncertainty estimation**, **probability calibration**, and **expected-cost decision theory**.

---

## 5. Methodology and Pipeline

```
dirty.csv
    │
    ▼
┌─────────────────────┐
│  Phase 1-2          │  Load & profile datasets
│  Data loading       │  Classify column types (NUMERICAL / BINARY /
│  & profiling        │  CATEGORICAL / MIXED / ID_LIKE)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Phase 3            │  Heuristic detection (no clean.csv):
│  Error detection    │  - Missing / null values
│                     │  - IQR statistical outliers
│                     │  - Rare / unknown categories
│                     │  - Format violations
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Phase 4            │  Mock LLM generates top-k repair candidates
│  Repair candidates  │  with probabilities for each detected cell.
│                     │  Strategies: frequency, similarity, statistical,
│                     │  range normalisation, format, whitespace.
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Phase 5            │  Three independent signals per cell:
│  Uncertainty        │  1. Raw confidence — top candidate probability
│  estimation         │  2. Self-consistency — agreement across N=5 seeds
│                     │  3. Entropy — H(p) = −Σ pᵢ log(pᵢ)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Phase 6            │  Labels from clean.csv (evaluation only).
│  Probability        │  Methods: Platt scaling (logistic) + isotonic
│  calibration        │  regression. Evaluated by Brier score and ECE.
│                     │  Reliability diagrams saved to results/figures/.
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Phase 7            │  For calibrated probability p:
│  Risk & cost        │  E[ACCEPT]  = (1−p) × C_wrong_repair
│  modeling           │  E[REVIEW]  = C_human_review
│                     │  E[LEAVE]   = C_leave_error
│                     │  → argmin selects the action
└──────────┬──────────┘
           │
    ┌──────┴──────┬──────────────┐
    ▼             ▼              ▼
ACCEPT_REPAIR  HUMAN_REVIEW  LEAVE_UNCHANGED
    │
    ▼
Evaluation vs clean.csv (empirical cost, top-1 accuracy)
```

### Calibrated Expected-Cost Decision

The central rule is analytical. Given cost parameters `C_wrong`, `C_review`, `C_leave` and calibrated probability `p`:

```
Accept preferred over Review when:  p > 1 − C_review / C_wrong
Accept preferred over Leave when:   p > 1 − C_leave  / C_wrong
```

With default costs (wrong=10, review=2, leave=5):

- Accept over Review: `p > 0.80`
- Accept over Leave: `p > 0.50`
- Final rule: **ACCEPT if p > 0.80**, else **HUMAN REVIEW** (since C_review < C_leave)

---

## 6. Project Phases

| Phase | Module | Status | Tests |
|---|---|---|---|
| 1 — Data Loading | `src/data/loader.py` | ✅ Complete | 33 |
| 2 — Dataset Profiling | `src/data/profiler.py`, `schema.py` | ✅ Complete | 66 |
| 3 — Error Detection | `src/detection/detector.py` | ✅ Complete | 54 |
| 4 — Mock LLM Repair | `src/repair/mock_repair.py` | ✅ Complete | 85 |
| 5 — Uncertainty Estimation | `src/uncertainty/estimator.py` | ✅ Complete | 74 |
| 6 — Probability Calibration | `src/calibration/calibrator.py` | ✅ Complete | 80 |
| 7 — Risk & Cost Modeling | `src/risk/cost_model.py` | ✅ Complete | 102 |
| 8 — Decision Policy | `src/decision/policy.py` | 🔲 Planned | — |
| 9 — Baselines | `src/baselines/baselines.py` | 🔲 Planned | — |
| 10 — Full Evaluation | `src/evaluation/` | 🔲 Planned | — |
| 11 — Experiment Runner | `run_experiment.py` | 🔲 Planned | — |
| 12 — Interactive Demo | `app.py` (Streamlit) | 🔲 Planned | — |

**Total tests passing: 494 across 7 complete phases.**

---

## 7. Technologies Used

| Technology | Role |
|---|---|
| Python 3.11 | Core language |
| pandas | Data loading, manipulation, profiling |
| numpy | Array operations, entropy, sensitivity grids |
| scikit-learn | Isotonic regression calibration |
| scipy | Logistic calibration (L-BFGS-B via `scipy.optimize.minimize`) |
| matplotlib | Reliability diagrams, Brier comparison charts |
| PyYAML | Cost configuration (`configs/costs.yaml`) |
| pytest | 494-test suite across 7 phases |

**No LLM API key is required.** The full pipeline runs offline using a deterministic seeded mock repairer.

---

## 8. Project Structure

```
.
├── configs/
│   ├── costs.yaml              # Per-dataset cost parameters (wrong/review/leave)
│   └── default.yaml            # Global project configuration
│
├── datasets/
│   ├── adult/                  # clean.csv, dirty.csv  (45,222 rows × 15 cols)
│   ├── breast_cancer/          # clean.csv, dirty.csv  (699 rows × 12 cols)
│   └── smartfactory/           # clean.csv, dirty.csv  (23,645 rows × 19 cols)
│
├── experiments/
│   └── experiment_config.yaml  # Baseline comparison experiment config
│
├── results/
│   └── figures/                # Reliability diagrams + Brier comparison (PNG)
│
├── src/
│   ├── data/
│   │   ├── loader.py           # DataLoader — CSV loading, column alignment
│   │   ├── profiler.py         # Column profiling, statistics, corruption summary
│   │   └── schema.py           # ColumnKind, DatasetProfile, CorruptionSummary, etc.
│   │
│   ├── detection/
│   │   ├── detector.py         # Detector — missing, outlier, format, category checks
│   │   └── evaluation.py       # DetectorEvaluator (uses clean.csv)
│   │
│   ├── repair/
│   │   ├── mock_repair.py      # MockRepairer — 7 strategies, seeded RNG
│   │   ├── prompts.py          # LLM prompt templates (for future real-LLM phase)
│   │   ├── evaluation.py       # RepairEvaluator (uses clean.csv)
│   │   └── llm_repair.py       # 🔲 Stub — real LLM integration (planned)
│   │
│   ├── uncertainty/
│   │   ├── estimator.py        # UncertaintyEstimator — confidence, consistency, entropy
│   │   └── evaluation.py       # UncertaintyCorrelationAnalysis (uses clean.csv)
│   │
│   ├── calibration/
│   │   ├── calibrator.py       # CalibrationBuilder, LogisticCalibrator, IsotonicCalibrator
│   │   ├── metrics.py          # Brier score, ECE, ECEBin
│   │   └── plots.py            # Reliability diagram, Brier comparison chart
│   │
│   ├── risk/
│   │   ├── cost_model.py       # Action, CostConfig, RiskCalculator, thresholds, sensitivity
│   │   └── evaluation.py       # RiskEvaluator (uses clean.csv), empirical cost report
│   │
│   ├── decision/
│   │   └── policy.py           # 🔲 Stub — decision policy (planned Phase 8)
│   ├── baselines/
│   │   └── baselines.py        # 🔲 Stub — baseline policies (planned Phase 9)
│   ├── evaluation/             # 🔲 Stubs — full evaluation metrics (planned Phase 10)
│   └── visualization/          # 🔲 Stub — additional plots (planned Phase 12)
│
├── tests/
│   ├── test_loader.py          # 33 tests — Phase 1
│   ├── test_profiler.py        # 66 tests — Phase 2
│   ├── test_detector.py        # 54 tests — Phase 3
│   ├── test_mock_repair.py     # 85 tests — Phase 4
│   ├── test_uncertainty.py     # 74 tests — Phase 5
│   ├── test_calibration.py     # 80 tests — Phase 6
│   ├── test_cost_model.py      # 102 tests — Phase 7
│   └── test_policy.py          # 🔲 Stub — Phase 8
│
├── demo_repair.py              # Phase 4 CLI demo
├── demo_uncertainty.py         # Phase 5 CLI demo
├── demo_calibration.py         # Phase 6 CLI demo
├── demo_risk.py                # Phase 7 CLI demo
│
├── app.py                      # 🔲 Stub — Streamlit interactive app (planned Phase 12)
├── run_experiment.py           # 🔲 Stub — experiment runner (planned Phase 11)
├── PROJECT_SPEC_FINAL.md       # Full build specification
├── pytest.ini                  # Test configuration
├── requirements.txt            # Python dependencies
└── .env.example                # Environment template (no real keys)
```

---

## 9. Installation

**Prerequisites:** Python 3.11+

```bash
# Clone the repository
git clone <repository-url>
cd DeepLearning_PROJ

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows

# Install dependencies
pip3 install -r requirements.txt
```

No API key is needed. The complete pipeline runs offline with the mock repairer.

---

## 10. How to Run

Each demo script exercises one complete phase end-to-end on all three datasets.

### Phase 4 — Mock Repair Candidates

```bash
python3 demo_repair.py
```

Shows top-3 repair candidates with probabilities for a sample of detected errors in each dataset. Evaluates top-1 and top-k accuracy against `clean.csv`.

### Phase 5 — Uncertainty Estimation

```bash
python3 demo_uncertainty.py
```

Computes raw confidence, self-consistency, and Shannon entropy for each repaired cell. Shows correlation of each signal with actual correctness.

### Phase 6 — Probability Calibration

```bash
python3 demo_calibration.py
```

Fits logistic (Platt scaling) and isotonic calibrators on each dataset. Prints calibration reports with Brier score and ECE. Saves reliability diagrams to `results/figures/`.

### Phase 7 — Risk and Cost Modeling

```bash
python3 demo_risk.py
```

Loads cost parameters from `configs/costs.yaml`, computes expected costs per action for every detected cell, shows the sensitivity sweep and analytical thresholds, and evaluates empirical cost of the policy against baselines.

### Run all demos in sequence

```bash
python3 demo_repair.py && \
python3 demo_uncertainty.py && \
python3 demo_calibration.py && \
python3 demo_risk.py
```

---

## 11. Running the Test Suite

```bash
# Run all 494 tests
python3 -m pytest

# Run a specific phase
python3 -m pytest tests/test_cost_model.py -v

# Run with short traceback on failure
python3 -m pytest --tb=short
```

**Current test count by phase:**

```
tests/test_loader.py        33 tests   Phase 1
tests/test_profiler.py      66 tests   Phase 2
tests/test_detector.py      54 tests   Phase 3
tests/test_mock_repair.py   85 tests   Phase 4
tests/test_uncertainty.py   74 tests   Phase 5
tests/test_calibration.py   80 tests   Phase 6
tests/test_cost_model.py   102 tests   Phase 7
─────────────────────────────────────
                           494 tests   TOTAL — all passing
```

---

## 12. Expected Outputs

### Console output (demo_calibration.py)

```
  DATASET: BREAST_CANCER  —  Phase 6 Calibration
  Calibration samples: 392  |  positive: 263 (67.1%)
  Method       Brier    ECE  Fitted
  raw         0.2191  0.1016  (baseline)
  logistic    0.2039  0.0556  yes
  isotonic    0.1899  0.0386  yes
```

### Console output (demo_risk.py)

```
  Analytical thresholds:
    ACCEPT > REVIEW when p > 0.9000
    ACCEPT > LEAVE  when p > 0.8000
    Policy: ACCEPT if p > 0.9000; otherwise HUMAN_REVIEW

  EMPIRICAL COST REPORT — BREAST_CANCER
  Strategy               Total cost  Mean/record
  Policy (risk model)       1615.00       4.5493
  Always ACCEPT             5800.00      16.3380
  Always REVIEW             1775.00       5.0000
  Always LEAVE              3550.00      10.0000
  Policy vs always-REVIEW : 160.00 units saves
  Policy vs always-LEAVE  : 1935.00 units saves
```

### Generated figures (`results/figures/`)

| File | Description |
|---|---|
| `reliability_breast_cancer.png` | Calibration curve — raw vs logistic vs isotonic |
| `reliability_smartfactory.png` | Calibration curve — SmartFactory sensors |
| `reliability_adult.png` | Calibration curve — Adult dataset (FP limitation documented) |
| `brier_comparison.png` | Bar chart comparing Brier scores across datasets and methods |

---

## 13. Configuration

### Cost parameters (`configs/costs.yaml`)

Costs are abstract **penalty units** (not monetary values). They are experimental assumptions defined by this project, not provided by the REIN benchmark.

| Dataset | `C_wrong_repair` | `C_human_review` | `C_leave_error` | Rationale |
|---|---|---|---|---|
| `adult` | 10 | 2 | 5 | Detections are statistical FPs; low domain stakes |
| `breast_cancer` | 50 | 5 | 10 | Healthcare domain — wrong repair has higher assumed consequence |
| `smartfactory` | 25 | 2 | 5 | Industrial sensors — wrong imputation propagates downstream |
| default | 10 | 2 | 5 | Used when no dataset-specific override exists |

To change cost assumptions, edit `configs/costs.yaml` — no code changes needed.

### Project defaults (`configs/default.yaml`)

Controls dataset paths, detection sensitivity (`iqr_k`, `zscore_threshold`), calibration split (`cal_fraction=0.30`), and number of self-consistency samples (`n_samples=5`).

---

## 14. Key Design Decisions and Limitations

### Data leakage boundary

`clean.csv` is accessed **only** in evaluation classes:
- `src/detection/evaluation.py` — detector precision/recall
- `src/repair/evaluation.py` — repair accuracy
- `src/uncertainty/evaluation.py` — signal-vs-correctness correlation
- `src/calibration/calibrator.py` — calibration labels only
- `src/risk/evaluation.py` — empirical cost report

All pipeline components (detection, repair, uncertainty, risk calculator) receive only `dirty_df`.

### Seeded deterministic mock repairer

Each cell's RNG seed is derived as `base_seed XOR MD5(dataset:column:row_id) % 2³¹`. This makes results fully reproducible and independent of iteration order.

### Adult dataset limitation

IQR outlier detection on `fnlwgt` produces statistical false positives. The true Adult corruption (income label flips) is undetectable from `dirty.csv` alone because the labels are indistinguishable from valid data. Calibrated probabilities collapse to ~0 for all Adult records. This is documented in `configs/costs.yaml` and printed by `demo_risk.py`.

### Breast Cancer ID columns

`Unnamed: 0` and `Sample code number` are classified as `ID_LIKE` and excluded from repair decisions (`skip_columns` in `compute_risk_batch`).

### Cost assumptions are illustrative

All cost values are abstract penalty units chosen to illustrate how the framework behaves under different domain assumptions. They do not represent real-world monetary values and are not provided by the REIN benchmark.

---

## 15. Conceptual Q&A

### Why do we need clean.csv AND dirty.csv?
`dirty.csv` is the input to the repair system. `clean.csv` provides ground truth to evaluate whether repairs were correct — it is never seen by the detection or repair components.

### What does error detection do?
It scans the dirty dataset for suspicious cells using heuristics: missing values, out-of-range numbers, malformed categories, format violations. It does not use `clean.csv`.

### What is confidence?
The probability the mock repairer assigns to its top repair candidate being correct. This is raw and uncalibrated — it reflects the repair strategy's internal scoring, not empirical correctness.

### What is entropy?
`H(p) = −Σ pᵢ log(pᵢ)`. High entropy means probability is spread across many candidates (uncertain). Low entropy means mass is concentrated on one candidate (certain). Normalised entropy `H / log(K)` scales to [0, 1].

### What is self-consistency?
Running the same repair request N=5 times with different random seeds and measuring how often the same answer appears. High agreement → stable; low agreement → uncertain.

### Why is calibration necessary?
Raw confidence can be systematically overconfident or underconfident. Calibration maps raw scores to empirically correct probabilities using a held-out labelled set (the calibration split, labelled using `clean.csv`).

### What does ECE measure?
Expected Calibration Error. Predictions are binned by confidence and the gap between mean confidence and actual accuracy in each bin is measured. Lower ECE = better calibrated.

### How is expected cost calculated?
`E[accept] = (1 − p) × C_wrong_repair`. If p is high, the repair is likely correct and the expected penalty is low. `E[review]` and `E[leave]` are constants that depend only on the cost configuration.

### Why is risk different from uncertainty?
Uncertainty measures how sure the model is. Risk combines uncertainty with the consequence of being wrong. A 70% confident repair on a low-cost field (`C_wrong=10`) can be ACCEPTed; the same confidence on a high-cost medical field (`C_wrong=50`) triggers HUMAN_REVIEW.

### What are the analytical thresholds?
For any cost configuration:
- ACCEPT beats REVIEW when `p > 1 − C_review / C_wrong`
- ACCEPT beats LEAVE when `p > 1 − C_leave / C_wrong`

These are derived analytically and match what the `argmin` would produce across all p values.

### Why not just use accuracy as the metric?
A policy maximising accuracy might apply many wrong repairs on high-cost fields, incurring large penalties. Total expected cost is a more complete objective that accounts for domain-level consequences.
