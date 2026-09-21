# Plan: make all three decisions live (calibrated abstention)

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked/finding

Branch: `feat/calibrated-abstention` (off `feat/phase11-dashboard`).
One commit per step. Every step ends with its tests passing and the full
suite still green with **no existing test edited** (that is the
backwards-compatibility check).

---

## 0. Problem, in fundamentals

Expected costs today:

```
E[AUTO]    = (1 - p) · C_wrong     falls with p
E[HUMAN]   = C_human               flat
E[ABSTAIN] = C_abstain             flat
```

Two flat lines never cross, so one of HUMAN / ABSTAIN is unreachable on
every dataset. No cost setting can fix that; it is structural.

What abstaining actually costs: nothing if the flagged cell was a false
alarm, `C_abstain` if it was genuinely corrupted. So

```
E[ABSTAIN] = q · C_abstain,   q = P(cell genuinely corrupted)
```

`q` must be a real probability. The detector's `detection_score` is not
one (Adult outliers score ~0.6 and 0% are corrupted). `q` is therefore
**learned from 0/1 corruption labels on a training split** — the same
permitted use of `clean_df` as Phase 6 (CLAUDE.md §3), symmetric with `p`.

Measured facts driving the design (2026-09-22, full datasets):

| Check (`detection_method`) | genuinely corrupted | datasets |
|---|---|---|
| missing_value, range_check, format_check, vocab_check | **100 %** | all |
| iqr_outlier (k=3) | **0 %** of 41,468 | smartfactory (if enabled), adult |
| iqr_outlier (k=1.5) on breast_cancer | **3 %** (5/195) | breast_cancer |

The outlier score bins carry no signal (2/38, 2/125, 1/3, 0/29). So `q`
is learned per check type; score bins are an ablation expected to add
nothing.

Where all three actions can legitimately appear in **one** dataset:
Breast Cancer with the Tukey fence k=1.5 (195 near-certain false alarms +
392 real errors, 52 of which have p > 0.9). Smart Factory's best repair
has p = 0.58, so AUTO never fires there — correct, report it. Adult is
100 % false alarms → all ABSTAIN, with **Sruthi's original
`c_leave_error = 5` restored**, because learned q ≈ 0 makes the hand-set
1 unnecessary.

Consistency requirement: if expected abstain cost is `q·C`, realised
abstain cost in evaluation must be `C·[cell was corrupted]`, otherwise the
evaluator penalises the policy for correctly ignoring false alarms.

---

## 1. Steps and todos

### Step 1 — Detection calibrator (new module, purely additive)
`src/calibration/detection_calibrator.py`

- [ ] 1.1 `DetectionLabelBuilder.build(detections, clean_df, dirty_df) -> List[int]`
      — the ONLY reader of `clean_df` here; label = dirty ≠ clean via the
      Phase 10 `ValueComparator` (NaN==NaN, numeric tolerance).
- [ ] 1.2 `DetectionCalibrator(group_by=("detection_method",), smoothing=1.0)`
      — per-group Beta(1,1)-smoothed corruption rate. `fit(records, labels)`,
      `predict(records) -> q`, `is_fitted()`. Unseen group → global rate.
      **Not fitted → q = 1.0** (today's implicit assumption: every flag is
      real). Optional score-bin grouping for the ablation.
- [ ] 1.3 `calibrate_detections(records, labels, seed, eval_fraction) ->
      DetectionCalibrationReport` with Brier/ECE of q vs raw
      `detection_score` on the hold-out, per-group table.
- [ ] 1.4 Tests `tests/test_detection_calibrator.py`: learned rates match
      empirical within smoothing; unseen group; unfitted → 1.0; empty;
      labels all-0 / all-1; determinism; no-leakage (fit/predict signatures,
      builder is sole clean reader); on real BC k=1.5 data q(iqr_outlier) < 0.1
      and q(missing_value) > 0.95.
- [ ] 1.5 Commit.

### Step 2 — Policy takes q (Phase 8, then Phase 7), default 1.0
- [ ] 2.1 `DecisionPolicy.decide(p, corruption_probability=1.0)`; validate
      q ∈ [0,1]; `e_abstain = q · c_abstain`; reason string shows
      `q × C_abstain`; `DecisionResult.corruption_probability: float = 1.0`
      appended LAST (frozen dataclass).
- [ ] 2.2 `decide_batch(probs, corruption_probabilities=None)`;
      `breakeven_auto_vs_abstain(corruption_probability=1.0)`.
- [ ] 2.3 Phase 7 mirror: `compute_expected_costs(p, corruption_probability=1.0)`,
      `ExpectedCosts.corruption_probability` (last, default 1.0),
      `compute_risk_record(..., corruption_probability=1.0)`,
      `compute_risk_batch(..., corruption_probs=None)`,
      `RiskRecord.corruption_probability` + `to_dict` key,
      `sensitivity_over_p(n_steps, corruption_probability=1.0)`.
      `analytical_thresholds()` docstring: note thresholds assume q=1.
- [ ] 2.4 `wrap_proposed_method` puts `corruption_probability` in
      `BaselineRecord.metadata`.
- [ ] 2.5 Tests (new file `tests/test_calibrated_abstention.py`, do not edit
      existing): q=1 reproduces old results on a p-grid exactly; q<1 scales
      abstain cost linearly; on a (p,q) grid with BC costs all three
      actions appear and regions are where the algebra says; q invalid
      rejected; ties at q·C == C_human resolve by documented priority;
      Phase 7 and 8 agree for the same (p,q).
- [ ] 2.6 Commit.

### Step 3 — Evaluation charges abstain only on real corruption
- [ ] 3.1 `CostEvaluator.evaluate`: ABSTAIN cost = `c_abstain` iff
      `not comparator.are_equal(rec.dirty_value, clean)`; add
      `CostReport.n_abstain_on_corrupted / n_abstain_on_clean` (default 0)
      and `to_dict` keys.
- [ ] 3.2 `RiskEvaluator` (Phase 7 demo path) same rule using `dirty_df`;
      "Always LEAVE" baseline = Σ c_leave·[corrupted].
- [ ] 3.3 Tests: abstain on clean cell → 0; on corrupted → c; existing
      all-abstain tests untouched and still pass (their cells are
      corrupted); Adult always-LEAVE == 0.
- [ ] 3.4 Commit. **Flag to Sruthi**: Phase 10 semantic change (only
      affects rows where abstain fires on a false alarm; BC/SF k=3 numbers
      unchanged).

### Step 4 — Runner + cost config
- [ ] 4.1 `run_pipeline(..., iqr_k=3.0, flag_outliers=None)` (None → per-
      dataset default as today). Bundle gains `detection_labels`,
      `corruption_probs`, `detection_calibration` (report), and
      `detection_method` per record.
- [ ] 4.2 `decide_all` passes q. `evaluate_policies` unchanged in interface.
- [ ] 4.3 Restore Adult `c_leave_error: 5` in `configs/costs.yaml`; rewrite
      the note (abstention now comes from learned q, not the config).
      Update `TestShippedCostConfigs` expectations — the ONE place an
      existing test is edited, because the config it pins is what changes.
      Add a test that Adult still all-abstains via q.
- [ ] 4.4 End-to-end tests: BC k=1.5 → all three actions present and
      iqr_outlier cells ≥ 95 % ABSTAIN, missing cells 0 % ABSTAIN;
      SF outliers-on → outliers ABSTAIN, missing HUMAN, AUTO = 0;
      BC k=3 → decisions identical to before this branch (regression pin).
- [ ] 4.5 Commit.

### Step 5 — Dashboard
- [ ] 5.1 Sidebar: "Detector" group — IQR fence k slider (1.0–3.0, default
      3.0) and "flag statistical outliers" toggle (default per dataset).
- [ ] 5.2 Section 4: add "P(cell genuinely corrupted) q" metric + how it was
      learned. Section 5: show `q × C_abstain`. Section 7: abstain line at
      the cell's q; new 2-D (p, q) decision-map figure; caption explains
      which action is unreachable and why, per current q.
- [ ] 5.3 `plots.decision_map(c_wrong, c_human, c_abstain, point=(p,q))`.
- [ ] 5.4 AppTest: BC with k=1.5 shows all three counts > 0; toggling
      outliers on SF makes ABSTAIN > 0; q metric present. Plot tests.
- [ ] 5.5 Commit.

### Step 6 — Honest experiment table (feeds Phase 13)
- [ ] 6.1 `experiments/three_actions_ablation.py` → markdown table:
      rows = {BC k=3, BC k=1.5, SF off, SF on, Adult} ×
      {constant abstain (old), calibrated abstain (new)};
      cols = action shares, realised cost, vs always-auto / threshold
      baselines; plus detection-calibration Brier (q vs raw score) and the
      score-bin ablation.
- [ ] 6.2 Write results into this file (§3) and CLAUDE.md §5.2.
- [ ] 6.3 Commit.

### Step 7 — Docs and handoff
- [ ] 7.1 CLAUDE.md: §2 formulas, §3 (second permitted clean_df reader),
      §4 state, §5.2 resolved. Memory note.
- [ ] 7.2 Message draft for Sruthi covering Steps 3.4 and 4.3.
- [ ] 7.3 Commit.

---

## 2. Decision points (need the user's yes/no before Step 3 / 4)

D1. Evaluator semantics change (Step 3): realised abstain cost only when
    the cell is truly corrupted. Recommended: yes — it is the only
    consistent choice once E[abstain] = q·C.
D2. Restore Adult `c_leave_error = 5` (Step 4.3). Recommended: yes — the
    learned q makes the hand-set 1 unnecessary and returns Sruthi's
    original config.
D3. Detector fence k=1.5 as an **experiment option** (default stays 3.0).
    Recommended: yes — reported as a detector-precision ablation, never as
    the default, so no existing numbers move.

## 3. Results (filled in at Step 6)

_(empty)_
