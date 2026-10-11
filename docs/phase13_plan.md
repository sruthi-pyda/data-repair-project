# Phase 13 — experiment runner, results, confidence intervals

Status: `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` finding

Branch: `feat/phase13-experiments` (off `feat/calibrated-abstention`).
One commit per step. Full suite green at every commit.

---

## 0. What this phase is for, in fundamentals

Right now every number in this project comes from a demo script or an
ad-hoc ablation, each with its own cap, its own seed handling and its own
cost arithmetic. That is not reproducible and it is not paper-ready.
Phase 13 makes **one command** regenerate **every** number and figure:

```bash
python run_experiment.py --dataset adult --mode mock
python run_experiment.py --all
```

Two deferred issues from CLAUDE.md §5.4 are in scope because they change
what the results *mean*, not just how they are produced:

1. **Single 70/30 split → repeated splits with confidence intervals.**
   Breast Cancer has ~392 calibration samples. One split gives one Brier
   number with no error bar, and we have been quoting differences of
   0.005 between methods. Without a CI those differences are not
   defensible.
2. **500-record demo cap → full datasets.** Acceptable for a demo, not
   for reported results.

### What varies, and what must not

The mock repairer is seeded from `(dataset, column, row_id)`, so the
*repairs* are a function of the pipeline seed. The calibration split is a
separate source of randomness. Conflating them would mean a "confidence
interval" that mixes two different things.

So: **`seed` controls repair/uncertainty noise; `split_seed` controls the
calibration split only.** Repeated-split CIs hold `seed` fixed and vary
`split_seed`. A separate (optional) sweep varies `seed` to show
sensitivity to mock-repair noise. They are reported as different things.

### Honest-reporting rules (inherited, non-negotiable)

- Costs are experimental penalty units. Never money.
- Adult's realised cost of 0 is *zero residual-corruption cost on the
  flagged cells*, always printed beside its 1,701 undetected corruptions.
- Mark every row where a calibrator fell back (Phase 6 base-rate, or
  q = 1).
- Report false negatives per dataset; the cost model prices them at zero.
- If the proposed method loses on a dataset, report it.
- No number may be hand-edited into a table. Everything regenerates.

---

## 1. API contract (fixed up front so steps can proceed in parallel)

```python
# src/pipeline/runner.py  — Step 1
run_pipeline(..., seed: int = 42, split_seed: Optional[int] = None)
    # split_seed=None  -> use `seed` (today's behaviour, bit-identical)
    # PipelineBundle gains `split_seed: int`

# src/experiments/aggregate.py  — Step 1 (new module)
@dataclass
class Interval:
    mean: float; lo: float; hi: float; std: float; n: int
    def to_dict(self) -> dict: ...
    def __str__(self) -> str:        # "0.1899 [0.1721, 0.2077]"

def bootstrap_interval(values, confidence=0.95, n_boot=10000, seed=42) -> Interval
def normal_interval(values, confidence=0.95) -> Interval      # t-based

@dataclass
class RepeatedSplitResult:
    dataset: str; n_splits: int; seed: int; split_seeds: List[int]
    brier: Dict[str, Interval]        # method -> interval  (raw/logistic/isotonic/multivariate)
    ece: Dict[str, Interval]
    n_fitted: Dict[str, int]          # how many splits that method actually fitted
    cost: Dict[str, Interval]         # policy name -> realised cost interval
    actions: Dict[str, Dict[str, Interval]]   # policy -> action -> share interval
    def to_dict(self) -> dict: ...

def repeated_splits(dataset, costs, n_splits=20, seed=42, base_split_seed=1000,
                    iqr_k=3.0, flag_outliers=None, n_records=0,
                    data_root="datasets") -> RepeatedSplitResult
```

`repeated_splits` runs the *pipeline* once per split seed (detection and
repair are unaffected by `split_seed`, but the calibrators and therefore
`p` and `q` are), then evaluates policies on each. Intervals are
bootstrap by default; `n_fitted` makes "this method fitted in 3 of 20
splits" visible instead of silently averaging fallbacks.

---

## 2. Steps

### Step 1 — split_seed + aggregation (`src/pipeline/runner.py`, new `src/experiments/`)
- [ ] 1.1 `split_seed` param; default None → `seed`; bundle field; used by
      both `_calibrate_all` and the detection calibrator. Default path
      bit-identical (assert on a p/q comparison test).
- [ ] 1.2 `src/experiments/aggregate.py` per the contract above.
- [ ] 1.3 Tests: interval maths against a hand-computed example and
      against scipy where available; degenerate inputs (n=1, all equal,
      empty); `n_fitted` counts; determinism; `split_seed=None` identical
      to today; different `split_seed` changes p but NOT detections or
      repairs (the isolation property this whole design rests on).
- [ ] 1.4 Commit.

### Step 2 — `run_experiment.py`
- [ ] 2.1 CLI: `--dataset {adult,breast_cancer,smartfactory}` | `--all`,
      `--mode {mock,real}` (real → not implemented, exit 2 with a clear
      message pointing at Phase 12), `--n-records 0`, `--seed 42`,
      `--splits 20`, `--iqr-k 3.0`, `--flag-outliers {auto,on,off}`,
      `--output-dir results`, `--quick` (small caps + 3 splits, for CI),
      `--no-figures`.
- [ ] 2.2 Writes, all regenerable, all deterministic:
      - `results/tables/*.md` and matching `*.csv`
      - `results/figures/*.png`
      - `results/predictions/<dataset>.csv` — one row per decided cell:
        row_id, column, detection_method, dirty_value, top_candidate,
        raw_confidence, p, q, E[auto], E[human], E[abstain], action,
        and (evaluation-only, clearly flagged) repair_correct,
        cell_genuinely_corrupted.
      - `results/run_metadata.json` — versions, seeds, git commit, UTC
        timestamp, per-dataset row/detection counts, wall time.
- [ ] 2.3 Four table groups (CLAUDE.md §6):
      **T1 repair quality** — detections, precision/recall/F1, top-1
      accuracy, false negatives, genuine corruptions.
      **T2 calibration** — raw vs logistic vs isotonic vs multivariate,
      Brier and ECE as mean [CI] over splits, with `n_fitted`.
      **T3 risk-aware performance** — action shares, realised cost,
      wrong-auto rate, abstentions on clean vs corrupted cells.
      **T4 policy comparison** — proposed vs always-auto vs always-human
      vs always-abstain vs threshold sweep, cost with CI.
- [ ] 2.4 Figures: reliability diagram per dataset, confidence
      distribution, cost comparison, action distribution, baseline vs
      proposed, and the (p,q) decision map with the population.
- [ ] 2.5 Tests (`tests/test_run_experiment.py`): `--quick --all` writes
      every expected artifact; byte-identical on re-run (determinism);
      `--mode real` exits 2 with the Phase 12 message; unknown dataset
      exits non-zero; predictions CSV has one row per decided cell and
      its ground-truth columns are the only ones flagged as such;
      metadata records the git commit; `--no-figures` skips PNGs.
- [ ] 2.6 Commit.

### Step 3 — run it for real and read the results
- [ ] 3.1 `python run_experiment.py --all` on full datasets, 20 splits.
- [ ] 3.2 Write `results/RESULTS.md`: the four tables, the figures, and
      an honest reading — what the numbers support, what they do not,
      where the method loses, and which rows are fallbacks.
- [ ] 3.3 Update CLAUDE.md §4 with the reproducible numbers and mark
      Phase 13 done.
- [ ] 3.4 Commit.

---

## 3. Results

_(filled in at Step 3)_
