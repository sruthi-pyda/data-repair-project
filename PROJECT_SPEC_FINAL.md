# Claude Build Specification
# Risk-Aware Human-in-the-Loop Framework for Trustworthy LLM-Assisted Data Repair

## 0. Starting condition

Build this project completely from scratch.

I currently have:
- No GitHub repository
- No existing codebase
- No virtual environment
- No API key
- An empty project folder
- Three REIN benchmark dataset folders that I will place inside the project:
  - `adult`
  - `breast_cancer`
  - `smartfactory`

The dataset folders contain the REIN clean/dirty data. Do NOT assume exact filenames or column names. Inspect the files and automatically identify the clean and dirty CSVs.

The project's research scope is based on this core idea:

> LLMs can generate useful data repairs but may be confidently wrong. Build a calibrated, cost-aware decision layer that decides whether to automatically apply a repair, send it to human review, or abstain, with the objective of reducing expected total repair cost rather than maximizing repair accuracy alone.

The implementation must be understandable to a data-science student and suitable for an academic project/viva.

---

# 1. Research objective

Implement and evaluate a risk-aware data-repair pipeline:

```text
Dirty Dataset
     |
     v
Error Detection
     |
     v
LLM Repair Candidate Generation
     |
     v
Uncertainty Estimation
(confidence + self-consistency + entropy)
     |
     v
Calibration
     |
     v
Risk / Expected Cost
     |
     v
Decision Policy
     |
     +-------------------+
     |         |         |
     v         v         v
   AUTO     HUMAN     ABSTAIN
  REPAIR    REVIEW
     |
     v
Evaluation
```

The central research contribution is NOT simply combining an LLM, uncertainty, and human review.

The important contribution is the **calibrated, cost-sensitive decision policy** that determines when an LLM repair should be trusted based on:
1. probability that the repair is correct,
2. calibration of that probability,
3. field-level risk/cost,
4. human-review cost,
5. abstention cost.

Use a transparent rule-based risk model as the baseline. A learned semantic risk estimator is out of scope unless added later as a clearly marked extension.

---

# 2. Datasets

Use these three datasets:

## Primary
`Adult`

Purpose:
- main end-to-end demonstration
- mixed categorical/numerical data
- intuitive data-quality examples
- primary Streamlit demo

## Secondary
`Breast Cancer`

Purpose:
- healthcare-domain experiment
- demonstrate why different fields/domains may justify different cost assumptions
- evaluate whether risk-aware decisions change under higher assumed consequences of wrong repairs

Do NOT claim that the benchmark itself provides clinical risk costs. Costs are experimental assumptions defined by this project.

## Tertiary
`Smart Factory`

Purpose:
- industrial-domain experiment
- numerical data
- test whether the framework generalizes beyond Adult

### Dataset handling requirements

Do not assume:
- exact CSV filenames
- exact column names
- exact datatypes
- exact error types
- exact number of rows

Inspect each dataset programmatically.

For every dataset, create a dataset metadata object containing:
- dataset name
- clean file path
- dirty file path
- row count
- column count
- column names
- inferred data types
- missing-value counts
- candidate corrupted-cell counts where ground truth is available

If the downloaded data does not have the expected clean/dirty pair, STOP and clearly report the detected structure instead of silently fabricating ground truth.

---

# 3. Critical ground-truth rule

The REIN clean data is ground truth for **evaluation**, not input to the repair model.

For every experiment:

```text
dirty.csv -> detection -> repair -> decision
clean.csv ---------------------------> evaluation only
```

Never use the clean value to generate an LLM repair candidate.

Ground truth can be used after prediction to determine:
- whether a dirty cell was actually corrupted
- whether the proposed repair is correct
- precision
- recall
- F1
- calibration correctness labels
- expected realized cost

Prevent data leakage.

---

# 4. Project structure

Create this structure:

```text
risk-aware-data-repair/
│
├── PROJECT_SPEC.md
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── app.py
├── run_experiment.py
│
├── configs/
│   ├── default.yaml
│   └── costs.yaml
│
├── data/
│   ├── adult/
│   ├── breast_cancer/
│   └── smartfactory/
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── profiler.py
│   │   └── schema.py
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   └── detector.py
│   │
│   ├── repair/
│   │   ├── __init__.py
│   │   ├── llm_repair.py
│   │   ├── mock_repair.py
│   │   └── prompts.py
│   │
│   ├── uncertainty/
│   │   ├── __init__.py
│   │   └── uncertainty.py
│   │
│   ├── calibration/
│   │   ├── __init__.py
│   │   └── calibrator.py
│   │
│   ├── risk/
│   │   ├── __init__.py
│   │   └── cost_model.py
│   │
│   ├── decision/
│   │   ├── __init__.py
│   │   └── policy.py
│   │
│   ├── baselines/
│   │   ├── __init__.py
│   │   └── baselines.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── repair_metrics.py
│   │   ├── calibration_metrics.py
│   │   └── cost_metrics.py
│   │
│   └── visualization/
│       ├── __init__.py
│       └── plots.py
│
├── experiments/
│   └── experiment_config.yaml
│
├── results/
│   ├── figures/
│   ├── tables/
│   └── predictions/
│
├── tests/
│   ├── test_loader.py
│   ├── test_uncertainty.py
│   ├── test_calibration.py
│   ├── test_cost_model.py
│   └── test_policy.py
│
└── notebooks/
    └── exploration.ipynb
```

Keep modules small and understandable.

---

# 5. Environment setup

Create:
- `requirements.txt`
- `.env.example`
- `.gitignore`

Use Python.

Recommended packages:
- pandas
- numpy
- scikit-learn
- scipy
- matplotlib
- seaborn
- streamlit
- pyyaml
- python-dotenv
- openai (only if the chosen implementation uses the OpenAI API)
- pytest
- plotly

Do not hard-code API keys.

The application must run in `MOCK_MODE=true` without any API key.

---

# 6. MOCK MODE IS REQUIRED

This is extremely important.

Before integrating a real LLM, build a deterministic mock LLM.

The mock mode must simulate:
- candidate repairs
- candidate probabilities
- correct and incorrect candidate generations
- uncertainty
- different confidence levels

It must allow the entire pipeline to run offline.

Example:

```text
Dirty:
occupation = "Adm-clericl"

Candidates:
Adm-clerical      0.82
Exec-managerial   0.12
Sales              0.06
```

The mock implementation must NOT cheat by directly reading the clean value during candidate generation.

It can use deterministic synthetic perturbation logic and seeded randomness.

The clean value may only be used later by evaluation code.

This allows the professor/demo to see the complete system without an API key.

---

# 7. Error detection

Create an error detection layer.

The detector should identify suspicious cells using information available in the dirty dataset, such as:

### Missing-value detection
Flag:
- NaN
- empty strings
- null-like strings

### Type/range anomaly detection
For numerical fields:
- impossible values
- extreme outliers using configurable statistical rules such as IQR/z-score where appropriate

### Categorical inconsistency detection
For categorical fields:
- rare/unseen values
- malformed categories
- whitespace/case inconsistencies
- values close to common valid categories using string similarity

### Format validation
For fields that clearly have structured formats:
- dates
- numeric strings
- identifiers
- percentages

Do NOT assume every rare value is an error.

Each detector should produce:

```python
{
    "row_id": ...,
    "column": ...,
    "dirty_value": ...,
    "error_type": ...,
    "detection_score": ...
}
```

The detector should be independent of the final clean value.

---

# 8. LLM repair candidate generation

For each detected cell, generate one or more candidate repairs.

The prompt should include:
- dataset/domain
- column name
- column datatype
- dirty value
- nearby/contextual row values
- representative valid values when available
- relevant schema constraints
- explicit instruction to return candidates and confidence

Do not send the entire dataset to the LLM unnecessarily.

Create a structured output such as:

```json
{
  "original_value": "Adm-clericl",
  "candidates": [
    {
      "value": "Adm-clerical",
      "confidence": 0.82
    },
    {
      "value": "Exec-managerial",
      "confidence": 0.12
    },
    {
      "value": "Sales",
      "confidence": 0.06
    }
  ]
}
```

Validate all LLM output before using it.

If an API is unavailable, automatically fall back to MOCK_MODE.

---

# 9. Uncertainty estimation

Implement at least these concepts:

## A. Top candidate confidence

Let:

```text
p = probability assigned to the top candidate
```

This is the raw confidence.

## B. Self-consistency

Run multiple independent repair attempts for the same corrupted value.

For example:

```text
Run 1 -> Adm-clerical
Run 2 -> Adm-clerical
Run 3 -> Exec-managerial
Run 4 -> Adm-clerical
Run 5 -> Adm-clerical
```

Frequency:

```text
Adm-clerical = 4/5 = 0.80
Exec-managerial = 1/5 = 0.20
```

Use the empirical frequency as one uncertainty signal.

## C. Entropy

For candidate probabilities:

```text
H(p) = - Σ p_i log(p_i)
```

Interpretation:
- low entropy = concentrated distribution = more certain
- high entropy = spread distribution = more uncertain

Normalize entropy when useful:

```text
H_normalized = H / log(K)
```

where K is the number of candidates.

Handle K=1 safely.

The implementation must clearly explain this in comments/docstrings.

---

# 10. Calibration

Raw LLM confidence is NOT assumed to be calibrated.

The system must include a calibration stage.

Use a separate calibration subset containing predictions and correctness labels.

Possible method:
- Platt scaling / logistic calibration
- isotonic regression

Prefer isotonic regression when enough calibration examples exist; otherwise use logistic/Platt scaling.

Input:

```text
raw confidence
```

Target:

```text
1 if repair is correct
0 if repair is incorrect
```

Output:

```text
calibrated probability of correctness
```

Example:

```text
Raw confidence       0.90
Actual correctness   0/1
Calibrated confidence 0.73
```

This is a key data-science component.

Do NOT calibrate on the same examples used to report final test performance.

Use a split such as:

```text
Detection/repair development data
        |
        +--> calibration set
        |
        +--> test set
```

If the dataset is small, use a documented train/calibration/test split and fixed random seed.

---

# 11. Calibration metrics

Implement:

## Expected Calibration Error (ECE)

Group predictions into confidence bins.

For each bin:

```text
|accuracy - average confidence|
```

Weight by number of samples.

Lower ECE is better.

## Brier Score

For binary correctness:

```text
Brier = mean((p - y)^2)
```

Lower is better.

## Negative Log-Likelihood

For binary correctness:

```text
NLL = -mean(y log(p) + (1-y) log(1-p))
```

Use clipping to avoid log(0).

Lower is better.

Also create a reliability diagram:

```text
Predicted confidence
vs
Observed accuracy
```

Include the ideal diagonal.

---

# 12. Risk / cost model

Use a transparent rule-based cost model.

Do NOT claim that REIN provides repair costs.

The project defines experimental costs.

Create configurable field/cost tiers, for example:

```yaml
costs:
  low:
    wrong_repair: 10
  medium:
    wrong_repair: 25
  high:
    wrong_repair: 100

human_review:
  default: 5

abstain:
  default: 2
```

Allow costs to be specified by:
- dataset
- column
- datatype
- field-risk tier

Example:

```yaml
adult:
  default_wrong_repair: 10

breast_cancer:
  default_wrong_repair: 50

smartfactory:
  default_wrong_repair: 25
```

These are experimental assumptions, not empirical real-world prices.

Document them clearly.

---

# 13. Expected cost decision policy

For a candidate repair with calibrated probability of correctness `p`:

```text
P(wrong) = 1 - p
```

For automatic repair:

```text
E[Cost_auto] = (1 - p) * C_wrong
```

Human review:

```text
E[Cost_human] = C_human
```

Abstention:

```text
E[Cost_abstain] = C_abstain
```

Choose:

```text
decision = argmin(
    E[Cost_auto],
    E[Cost_human],
    E[Cost_abstain]
)
```

The system should support:

```text
AUTO_REPAIR
HUMAN_REVIEW
ABSTAIN
```

This is the central decision layer.

---

# 14. Important distinction: uncertainty vs risk

The UI and README must explicitly explain:

### Uncertainty

“How sure is the model that this repair is correct?”

Measured using:
- confidence
- self-consistency
- entropy
- calibration

### Risk

“How expensive would it be if this repair were wrong?”

Depends on:
- probability of being wrong
- wrong-repair cost
- human-review cost
- abstention cost

Therefore:

```text
High confidence != automatically safe
```

A highly confident repair can still be rejected when its expected wrong-repair cost is high.

---

# 15. Baselines

Implement at least these baselines.

## Baseline 1: Always Auto-Repair

Always choose the top LLM candidate.

This tests:

> What happens if we trust the LLM every time?

## Baseline 2: Confidence-Only Abstention

Use a fixed confidence threshold.

Example:

```text
if confidence >= 0.8:
    AUTO
else:
    HUMAN_REVIEW
```

Evaluate several thresholds rather than assuming 0.8 is optimal.

This tests:

> Is confidence alone enough?

## Proposed Method: Calibrated Risk-Aware Policy

Use:

```text
calibrated probability
+
field-level cost
+
human cost
+
abstention cost
```

Then choose the minimum expected cost action.

The comparison should demonstrate that the proposed method is not merely maximizing F1.

---

# 16. Evaluation metrics

Evaluate three categories.

## A. Repair quality

Calculate:
- precision
- recall
- F1
- number of repaired cells
- number of correct repairs
- number of incorrect repairs

For categorical values, exact normalized matching may be appropriate.

For numerical values, define a documented tolerance where appropriate.

Do not blindly use exact equality for floating-point repairs.

## B. Calibration

Calculate:
- ECE
- Brier score
- NLL

Compare:
- raw confidence
- calibrated confidence

## C. Risk-aware evaluation

Calculate:

```text
Total Cost =
Σ wrong-auto-repair costs
+ Σ human-review costs
+ Σ abstention costs
```

Also report:
- average cost per detected error
- auto-repair rate
- human-review rate
- abstention rate
- wrong-auto-repair rate

The main objective is:

> minimize total expected/realized cost while maintaining acceptable repair quality.

---

# 17. Streamlit visualization

Build a small but polished practical demonstration.

The visualization is important because I need to show the professor how the research works in practice.

The app should have:

## Sidebar

- Dataset selector:
  - Adult
  - Breast Cancer
  - Smart Factory
- Mode:
  - Mock
  - Real LLM
- Risk/cost controls
- Confidence threshold for baseline
- Number of repair samples
- Random seed

## Main dashboard

### Section 1: Dataset overview

Show:
- rows
- columns
- missing values
- detected errors

### Section 2: Dirty record

Show an example suspicious row/cell:

```text
Column: occupation
Dirty value: Adm-clericl
```

### Section 3: Candidate repairs

Display:

| Candidate | Raw Probability |
|---|---:|
| Adm-clerical | 0.82 |
| Exec-managerial | 0.12 |
| Sales | 0.06 |

### Section 4: Uncertainty

Display:
- raw confidence
- self-consistency
- entropy
- calibrated confidence

### Section 5: Expected cost

Display:

```text
Auto-repair expected cost: ₹X / units
Human review cost:         ₹Y / units
Abstain cost:              ₹Z / units
```

Do not imply these are real monetary costs. Label them as experimental cost units.

### Section 6: Final decision

Large visible result:

```text
AUTO-REPAIR
```

or:

```text
HUMAN REVIEW
```

or:

```text
ABSTAIN
```

Also explain WHY:

```text
Calibrated probability of correctness = 0.79

Expected auto cost = 4.2
Human cost = 5
Abstention cost = 2

Decision = ABSTAIN
```

### Section 7: Interactive risk demonstration

Include sliders for:
- wrong-repair cost
- human-review cost
- abstention cost

When the user changes a cost, recompute the decision live.

This should visually demonstrate the project's core idea:

> The same model confidence can produce different decisions when the consequence of being wrong changes.

### Section 8: Evaluation plots

Show:
- confidence histogram
- reliability diagram
- cost comparison
- action distribution
- repair quality comparison between baselines and proposed method

Keep the dashboard small enough to run smoothly on a laptop.

---

# 18. Experiment runner

Create:

```bash
python run_experiment.py
```

It should:
1. load selected dataset
2. profile it
3. detect errors
4. generate repairs
5. calculate uncertainty
6. calibrate
7. calculate expected costs
8. apply decisions
9. evaluate
10. save predictions/results
11. save plots

Allow configuration through YAML/CLI.

Example:

```bash
python run_experiment.py --dataset adult --mode mock
python run_experiment.py --dataset breast_cancer --mode mock
python run_experiment.py --dataset smartfactory --mode mock
```

---

# 19. Reproducibility

Use:
- fixed random seeds
- deterministic mock mode
- saved experiment configuration
- saved predictions
- saved metrics
- saved plots

Save a JSON/YAML metadata file with every experiment:
- dataset
- seed
- model/mode
- cost settings
- calibration method
- split sizes
- timestamp
- metrics

---

# 20. Testing

Write unit tests for:
- dataset discovery
- CSV loading
- missing-value detection
- uncertainty calculation
- entropy calculation
- calibration
- ECE
- Brier score
- NLL
- expected-cost calculation
- decision policy

At minimum verify:

```text
If p is high and wrong-repair cost is low:
    AUTO may be selected.

If p is low:
    HUMAN or ABSTAIN should be selected depending on costs.

If wrong-repair cost increases:
    AUTO should become less attractive.

If human cost decreases:
    HUMAN_REVIEW should become more attractive.
```

---

# 21. README requirements

Write a clear README containing:

1. Project motivation
2. Research question
3. Dataset description
4. Architecture
5. Data-science concepts
6. Installation
7. Dataset placement
8. Mock-mode usage
9. Real-LLM setup
10. Experiment commands
11. Streamlit command
12. Evaluation metrics
13. Limitations
14. Reproducibility instructions

Installation should be simple:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows instructions may also be included.

Run demo:

```bash
streamlit run app.py
```

---

# 22. Academic honesty / limitations

The README and comments must clearly distinguish:

### What the framework actually demonstrates
- LLM-assisted candidate repair
- uncertainty estimation
- confidence calibration
- explicit risk/cost modeling
- cost-sensitive routing
- human-in-the-loop decision making

### What it does NOT claim
- perfect error detection
- perfect LLM confidence
- real-world clinical/industrial monetary costs
- that the REIN benchmark provides repair-cost labels
- that the risk model is learned from real business consequences

State that field-level costs are experimental assumptions used to test the decision framework.

---

# 23. Implementation order

Do NOT build everything at once.

Build and test in this exact order:

## Phase 1
Create project structure and environment.

## Phase 2
Load and profile Adult, Breast Cancer, Smart Factory.

## Phase 3
Implement error detection.

## Phase 4
Implement deterministic mock repair.

## Phase 5
Implement confidence, self-consistency and entropy.

## Phase 6
Implement calibration.

## Phase 7
Implement rule-based cost model.

## Phase 8
Implement AUTO/HUMAN/ABSTAIN policy.

## Phase 9
Implement baselines.

## Phase 10
Implement evaluation.

## Phase 11
Implement Streamlit dashboard.

## Phase 12
Integrate real LLM API.

## Phase 13
Run final experiments and generate tables/plots.

At the end of each phase, run tests and make sure the previous phase still works.

---

# 24. Critical implementation constraints

1. Do not use clean ground truth during candidate generation.
2. Do not hard-code dataset-specific clean values.
3. Do not fabricate results.
4. Do not fabricate dataset statistics.
5. Do not claim costs are supplied by REIN.
6. Do not require an API key to run the project.
7. Do not require PostgreSQL or the full REIN framework just to run our experiments unless genuinely necessary.
8. Prefer pandas/NumPy/scikit-learn for the core implementation.
9. Keep the code modular and readable.
10. Use type hints where practical.
11. Add docstrings to data-science functions.
12. Explain formulas in comments/docstrings.
13. Handle missing/invalid LLM responses safely.
14. Log experiment configuration.
15. Never silently swallow exceptions.
16. Never overwrite raw datasets.
17. Never modify `clean.csv` or `dirty.csv`.
18. Use a fixed random seed for the mock experiments.
19. Make all thresholds and costs configurable.
20. Make the Streamlit demo work completely offline in mock mode.

---

# 25. Final expected result

When finished, I should be able to run:

```bash
streamlit run app.py
```

and see a working practical demonstration.

I should also be able to run:

```bash
python run_experiment.py --dataset adult --mode mock
```

and obtain:
- repair predictions
- confidence/uncertainty values
- calibrated probabilities
- decisions
- repair metrics
- calibration metrics
- total cost
- plots
- saved result tables

The final project must demonstrate the following research story clearly:

```text
LLM can repair data
        ↓
but LLM confidence can be unreliable
        ↓
estimate uncertainty
        ↓
calibrate confidence
        ↓
combine calibrated probability with repair risk
        ↓
compare expected costs
        ↓
AUTO / HUMAN / ABSTAIN
        ↓
evaluate accuracy + calibration + total cost
```

The project should be designed so that I can explain every major component in a viva, especially:
- why clean and dirty data are both needed
- what error detection does
- how an LLM generates repair candidates
- what confidence means
- what entropy means
- what self-consistency means
- why calibration is necessary
- what ECE/Brier/NLL measure
- how expected repair cost is calculated
- why risk is different from uncertainty
- how the decision policy works
- why baselines are necessary
- why accuracy alone is insufficient
- how the Streamlit visualization demonstrates the research contribution.

Before declaring the project complete, provide a final `README.md` section called:

## Viva Explanation Guide

with short, accurate explanations of every major data-science concept and implementation decision.
