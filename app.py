"""
app.py — Phase 11 Streamlit dashboard
=====================================
Interactive demonstration of the research claim:

    A calibrated probability of correctness, combined with explicit costs of
    being wrong, produces better decisions than acting on raw confidence.
    The same confidence should yield different actions when the consequences
    change.

Run with:

    streamlit run app.py

Works fully offline in mock mode; no API key is needed.

REAL LLM MODE (Phase 12)
------------------------
"Real LLM" in the sidebar switches the pipeline to
``src/repair/llm_repair.py``. It needs two things, and says which one is
missing:

1. a usable key — ``LLM_API_KEY`` (or ``LLM_API_KEYS`` for the rotating
   pool) in a gitignored ``.env``; and
2. the **"Send real API calls"** box ticked.

The second gate is deliberate. Dragging a radio button must not spend API
quota, and a dashboard rerun re-runs the pipeline: without an explicit arm,
every slider nudge could become hundreds of billed calls. Unarmed, the app
falls back to mock repair, says so, and makes no API call. When armed,
Sections 3 and 4 name the model and print the fallback count and rate,
because a real-mode run in which most cells fell back is a mock run with a
label on it.

No API key is read into the page, a cache key, a log line or a figure; only
the model id, the pool size and the counters are displayed.

LAYOUT
------
Sidebar   dataset · mode · detector knobs (IQR fence k, outlier flagging) ·
          three cost sliders · baseline threshold · number of records ·
          random seed
Sections  1 dataset overview
          2 one dirty record
          3 repair candidates
          4 uncertainty signals  (p and q, and how q was learned)
          5 expected costs (experimental cost units), E[ABSTAIN] = q·C_abstain
          6 final decision
          7 interactive risk demonstration  ← the centrepiece
              cost curves at this cell's q, plus the derived (p, q) decision map
          8 baseline comparison  (ground truth used here, and only here)
          9 plots

LEAKAGE RULE (CLAUDE.md §3)
---------------------------
Sections 1–7 never display a clean value next to a dirty cell. The clean
value appears only in Section 8, which is visibly headed as evaluation.
The pipeline stages are in ``src/pipeline/runner.py``; ``decide_all`` has no
ground-truth parameter by construction.

Two detection-side quantities must not be confused:

* ``bundle.corruption_probs[i]`` — ``q``, the calibrated P(cell genuinely
  corrupted). A **decision input**, learned per detector check type from a
  train split, so it may be displayed anywhere (Sections 4, 5 and 7 do).
* ``bundle.detection_labels[i]`` — the 0/1 ground-truth corruption label.
  **Evaluation only**, and appears nowhere before Section 8.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st

from src.decision.policy import DecisionAction, DecisionResult
from src.pipeline import (
    DATASETS,
    PipelineBundle,
    PolicyEvaluation,
    decide_all,
    default_flag_outliers,
    evaluate_policies,
    run_pipeline,
)
from src.repair.llm_repair import LLMConfig
from src.risk.cost_model import CostConfig
from src.visualization import plots as P

COSTS_YAML = Path("configs/costs.yaml")
COST_UNITS = P.COST_UNITS_LABEL

DATASET_BLURB = {
    "breast_cancer": (
        "Wisconsin Breast Cancer (699 rows). Corruption: missing tokens, "
        "format errors (`1a0`, `??`) and outliers in cytology scores. "
        "Flagship demo: detection and repair both work, so calibration has signal."
    ),
    "smartfactory": (
        "Smart Factory sensor data (subset). Corruption: injected missing "
        "values in sensor columns. Repairs are median imputations, which are "
        "often wrong, so the policy should route most cells to a human."
    ),
    "adult": (
        "UCI Adult census. Corruption: income-label flips (valid value → valid "
        "value) that are **undetectable from dirty data alone**. Detection finds "
        "only statistical false positives, so this is a deliberate negative "
        "result: the right action is to leave the flagged cells alone."
    ),
}

ACTION_STYLE = {
    DecisionAction.AUTO_REPAIR: ("✅", "success", "Apply the top candidate automatically."),
    DecisionAction.HUMAN_REVIEW: ("🧑‍⚕️", "warning", "Send this cell to a reviewer."),
    DecisionAction.ABSTAIN: ("⏸️", "info", "Leave the cell unchanged."),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Cached pipeline
# ═══════════════════════════════════════════════════════════════════════════════


@st.cache_resource(show_spinner="Running detection → repair → uncertainty → calibration …")
def load_bundle(dataset: str, n_records: int, seed: int, n_samples: int,
                iqr_k: float, flag_outliers: bool,
                repair_mode: str = "mock") -> PipelineBundle:
    """``iqr_k``, ``flag_outliers`` and ``repair_mode`` are part of the cache
    key: changing a detector knob re-runs detection and switching the repairer
    re-runs the repairs, so the cached bundle can never be stale with respect
    to the sidebar. Only the *mode string* is in the key — never a config
    object, and so never a key."""
    return run_pipeline(
        dataset, n_records=n_records, seed=seed, n_samples=n_samples,
        iqr_k=iqr_k, flag_outliers=flag_outliers, mode=repair_mode,
    )


def llm_status() -> dict:
    """
    What the sidebar may show about the LLM configuration.

    Reads `.env` / ``os.environ`` through ``LLMConfig.from_env`` and returns
    only the model id, the pool size and whether a usable key exists. The key
    itself never leaves this function.
    """
    config = LLMConfig.from_env()
    return {
        "configured": config.is_configured,
        "model": config.model,
        "n_keys": config.n_api_keys,
        "base_url": config.base_url,
    }


def llm_run_note(b: PipelineBundle) -> Optional[str]:
    """
    One line of real-mode provenance, or ``None`` in mock mode.

    States the model, how many cells the model actually answered, and how
    many fell back to the mock repairer — the number that decides whether
    anything on the page is an LLM result at all.
    """
    if getattr(b, "repair_mode", "mock") != "real":
        return None
    stats = getattr(b, "llm_stats", None) or {}
    rate = float(stats.get("fallback_rate", 0.0) or 0.0)
    return (
        f"**Real LLM mode** · model `{stats.get('model', '?')}` · "
        f"{int(stats.get('successes', 0) or 0):,} of "
        f"{int(stats.get('cells', 0) or 0):,} cells answered by the model · "
        f"**{int(stats.get('fallbacks', 0) or 0):,} fell back to mock repair "
        f"({rate:.1%})** · {int(stats.get('cache_hits', 0) or 0):,} cache hits "
        f"· {int(stats.get('rate_limited', 0) or 0):,} rate limits · "
        f"{int(stats.get('key_rotations', 0) or 0):,} key rotations."
    )


def make_costs(dataset: str, c_wrong: float, c_human: float, c_abstain: float) -> CostConfig:
    return CostConfig(
        c_wrong_repair=float(c_wrong),
        c_human_review=float(c_human),
        c_leave_error=float(c_abstain),
        dataset_name=dataset,
    )


def yaml_defaults(dataset: str) -> CostConfig:
    try:
        return CostConfig.from_yaml(COSTS_YAML, dataset)
    except Exception:  # pragma: no cover - config missing in odd deployments
        return CostConfig.default(dataset)


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════


def sidebar() -> dict:
    st.sidebar.title("Controls")

    dataset = st.sidebar.selectbox("Dataset", DATASETS, index=0, key="dataset")

    mode = st.sidebar.radio("Repair mode", ["Mock (offline)", "Real LLM"], index=0, key="mode")
    repair_mode = "mock"
    if mode == "Real LLM":
        status = llm_status()
        if not status["configured"]:
            st.sidebar.warning(
                "Phase 12 real-LLM repair (`src/repair/llm_repair.py`) is "
                "implemented, but no usable API key is configured — falling "
                "back to mock repair. No API call is made.\n\n"
                "Put `LLM_API_KEY=<your key>` in `.env` in the repository "
                "root (or `LLM_API_KEYS=<key1,key2,…>` for the rotating pool, "
                "plus `LLM_BASE_URL` / `LLM_MODEL` for a non-OpenAI "
                "provider). `.env` is gitignored and this app never prints, "
                "logs or caches a key."
            )
        elif st.sidebar.checkbox(
            "Send real API calls (spends quota)", value=False, key="llm_live",
            help=(
                "Off by default on purpose: every rerun of this page re-runs "
                "the pipeline, so one slider nudge could otherwise become "
                "hundreds of billed model calls."
            ),
        ):
            repair_mode = "real"
            st.sidebar.success(
                f"Real LLM repair armed · model `{status['model']}` · "
                f"{status['n_keys']} key(s) in the rotation pool. Cells that "
                f"fail fall back to mock repair per cell; the count is shown "
                f"in Sections 3 and 4."
            )
        else:
            st.sidebar.warning(
                f"Phase 12 real-LLM repair is available (model "
                f"`{status['model']}`) but **not armed** — falling back to "
                f"mock repair. No API call is made. Tick *Send real API "
                f"calls* to use the model."
            )

    st.sidebar.subheader("Detector")
    iqr_k = st.sidebar.slider(
        "IQR fence k (Tukey multiplier)", 1.0, 3.0, 3.0, 0.5, key=f"iqr_k_{dataset}",
        help=(
            "Lower k = tighter fence = many more cells flagged as statistical "
            "outliers. On these datasets almost all of those extra flags are "
            "false alarms, which is exactly what the learned q measures."
        ),
    )
    flag_outliers = st.sidebar.checkbox(
        "Flag statistical outliers", value=default_flag_outliers(dataset),
        key=f"flag_outliers_{dataset}",
        help=(
            "Off for Smart Factory by default (configs/default.yaml): its "
            "non-missing sensor values are known valid, so the IQR check can "
            "only produce false positives."
        ),
    )
    st.sidebar.caption(
        "`breast_cancer` with k=1.5 is the configuration where **all three "
        "actions occur**: 195 near-certain false alarms alongside real errors."
    )

    defaults = yaml_defaults(dataset)
    st.sidebar.subheader(f"Costs ({COST_UNITS})")
    st.sidebar.caption(
        "Abstract penalty weights from `configs/costs.yaml`. Not money, not "
        "clinical harm, not supplied by the benchmark."
    )
    # Keys include the dataset so switching datasets resets sliders to that
    # dataset's YAML values instead of carrying the previous dataset's over.
    c_wrong = st.sidebar.slider(
        "C_wrong — applying an incorrect repair", 1.0, 100.0,
        float(defaults.c_wrong_repair), 1.0, key=f"c_wrong_{dataset}",
    )
    c_human = st.sidebar.slider(
        "C_human — sending a cell to review", 0.0, 50.0,
        float(defaults.c_human_review), 0.5, key=f"c_human_{dataset}",
    )
    c_abstain = st.sidebar.slider(
        "C_abstain — leaving the cell unchanged", 0.0, 50.0,
        float(defaults.c_leave_error), 0.5, key=f"c_abstain_{dataset}",
    )

    st.sidebar.subheader("Baseline")
    threshold = st.sidebar.slider(
        "Raw-confidence threshold for auto-repair", 0.5, 0.95, 0.8, 0.05, key="threshold",
    )

    st.sidebar.subheader("Pipeline")
    n_records = st.sidebar.slider("Detected cells to carry forward", 50, 1000, 300, 50, key="n_records")
    n_samples = st.sidebar.slider("Self-consistency samples", 1, 10, 5, 1, key="n_samples")
    seed = st.sidebar.number_input("Random seed", 0, 10_000, 42, 1, key="seed")

    return dict(
        dataset=dataset, mode=mode, repair_mode=repair_mode,
        c_wrong=c_wrong, c_human=c_human,
        c_abstain=c_abstain, threshold=threshold, n_records=int(n_records),
        n_samples=int(n_samples), seed=int(seed), iqr_k=float(iqr_k),
        flag_outliers=bool(flag_outliers),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Sections
# ═══════════════════════════════════════════════════════════════════════════════


def section_overview(b: PipelineBundle) -> None:
    st.header("1 · Dataset overview")
    st.markdown(DATASET_BLURB.get(b.dataset, ""))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{b.profile.n_rows:,}")
    c2.metric("Columns", b.profile.n_cols)
    c3.metric("Cells flagged by detector", f"{b.n_detected_total:,}")
    c4.metric("Carried into demo", f"{b.n_records:,}")

    kinds = pd.DataFrame(
        [
            {
                "column": cp.name,
                "kind": cp.kind.value,
                "dtype (dirty)": cp.dtype_dirty,
                "missing in dirty": cp.missing_dirty,
            }
            for cp in b.profile.columns.values()
        ]
    )
    with st.expander("Column profile (population-level schema knowledge)"):
        st.dataframe(kinds, use_container_width=True, hide_index=True)
    with st.expander("Dirty data sample (first 10 rows)"):
        st.dataframe(b.dirty_df.head(10), use_container_width=True)


def section_record(b: PipelineBundle) -> int:
    st.header("2 · One dirty record")
    idx_options = b.decision_indices() or list(range(b.n_records))
    if not idx_options:
        st.warning("No detections for this dataset with the current settings.")
        st.stop()

    def _label(i: int) -> str:
        r = b.unc_records[i]
        return f"row {r.row_id} · {r.column} · {r.error_type}"

    i = st.selectbox(
        "Pick a detected cell", idx_options, format_func=_label, key=f"record_{b.dataset}",
    )
    det = b.detections[i]
    rec = b.unc_records[i]

    left, right = st.columns([2, 3])
    with left:
        st.markdown(f"**Column:** `{rec.column}`")
        st.markdown(f"**Dirty value:** `{det.dirty_value!r}`")
        st.markdown(f"**Detected as:** {det.error_type.value if hasattr(det.error_type, 'value') else det.error_type}")
        st.markdown(f"**Detector reason:** {det.reason}")
        st.markdown(f"**Detection score:** {det.detection_score:.2f}")
    with right:
        st.caption("Row context as the repairer sees it (dirty data only)")
        row = b.dirty_df.iloc[[rec.row_id]].T.astype(str)
        row.columns = [f"row {rec.row_id}"]
        st.dataframe(row, use_container_width=True, height=260)
    return i


def section_candidates(b: PipelineBundle, i: int) -> None:
    st.header("3 · Repair candidates")
    rr = b.repair_results[i]
    note = llm_run_note(b)
    if note is None:
        st.caption(f"Repair method: `{rr.repair_method}` (mock repairer — not an LLM)")
    else:
        # `repair_method` is 'llm' or 'llm->mock (<reason>)', so this cell's
        # own provenance is visible beside the run-level counters.
        st.caption(f"Repair method for this cell: `{rr.repair_method}`")
        st.info(note)
    table = pd.DataFrame(
        [{"rank": k + 1, "candidate": str(c.value), "probability": c.probability}
         for k, c in enumerate(rr.candidates)]
    )
    c1, c2 = st.columns([2, 3])
    c1.dataframe(table, hide_index=True, use_container_width=True)
    c2.pyplot(P.candidate_bar([(c.value, c.probability) for c in rr.candidates]),
              use_container_width=True)


def _q_of(b: PipelineBundle, i: int) -> float:
    """``q = P(cell genuinely corrupted)`` for record ``i``.

    A decision input, not ground truth: it is the calibrated rate of the
    detector check that flagged the cell, so it may be shown anywhere.
    ``bundle.detection_labels`` is the ground-truth counterpart and appears
    only in Section 8.
    """
    qs = b.corruption_probs
    return float(qs[i]) if i < len(qs) else 1.0


def _group_rate_str(b: PipelineBundle, method: str) -> str:
    rep = b.detection_calibration
    if rep is None:
        return "not available"
    for g in rep.groups:
        if method in tuple(str(x) for x in g.get("group", ())):
            return f"{g['rate']:.3f} ({g['n_corrupted']}/{g['n']} on the train split)"
    return "unseen group → global rate"


def _detection_quality_expander(b: PipelineBundle) -> None:
    rep = b.detection_calibration
    if rep is None:
        return
    with st.expander("Detection calibration quality — how q was fitted"):
        if rep.groups:
            st.table(pd.DataFrame([
                {
                    "detector check (group)": " · ".join(str(x) for x in g["group"]),
                    "n (train split)": g["n"],
                    "n corrupted (train split)": g["n_corrupted"],
                    "learned rate q": f"{g['rate']:.3f}",
                }
                for g in rep.groups
            ]))
            st.caption(
                "**`learned rate q` is a group-level rate fitted on the train "
                "split — not a verdict on any individual cell.** A cell in a "
                "group with q = 0.04 may still be corrupted; the policy is "
                "betting on the rate, which is the whole point of calibration."
            )
        else:
            st.caption("No groups fitted; q falls back to 1.0 for every record.")
        st.table(pd.DataFrame({
            "method": ["raw detection_score", "calibrated q"],
            "Brier": [rep.metric_str("raw", "brier"), rep.metric_str("calibrated", "brier")],
            "ECE": [rep.metric_str("raw", "ece"), rep.metric_str("calibrated", "ece")],
        }))
        st.caption(
            f"Hold-out of {rep.n_eval} flagged cells (base rate "
            f"{rep.base_rate:.3f}); lower is better. 'raw' treats the "
            "detector's suspicion score as if it were a probability — it is "
            "not one, which is why q is learned instead."
        )


def section_uncertainty(b: PipelineBundle, i: int) -> None:
    st.header("4 · Uncertainty signals")
    rec = b.unc_records[i]
    p_cal = b.calibrated_probs[i]
    q = _q_of(b, i)
    method = b.detection_methods[i] if i < len(b.detection_methods) else "unknown"
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Raw confidence", f"{rec.raw_confidence:.3f}")
    c2.metric("Self-consistency", f"{rec.self_consistency:.2f}",
              help=f"Agreement across {rec.n_samples} resampled repair calls")
    c3.metric("Normalised entropy", f"{rec.normalized_entropy:.3f}",
              help="0 = one candidate dominates, 1 = uniform over candidates")
    c4.metric("Calibrated p(correct)", f"{p_cal:.3f}",
              delta=f"{p_cal - rec.raw_confidence:+.3f} vs raw",
              help=f"Calibrator: {b.calibration_method}")
    c5.metric("q = P(cell genuinely corrupted)", f"{q:.3f}",
              help=(
                  f"Learned for the `{method}` check type; that group's "
                  f"learned corruption rate is {_group_rate_str(b, method)}. "
                  "q is a decision input, not a verdict on this cell."
              ))
    st.caption(
        "**How q is learned.** Detections are grouped by detector check type "
        "(`detection_method`), and each group's corruption rate is fitted on a "
        "train split of 0/1 corruption labels, then predicted for every cell — "
        "the same permitted use of ground-truth labels that produces p "
        "(CLAUDE.md §3). A missing-value flag is nearly always a real error "
        "(q → 1); an IQR outlier on these datasets nearly never is (q → 0). "
        "If the calibrator cannot fit, q = 1.0 everywhere, which reproduces "
        "the pre-calibration policy exactly."
    )
    note = llm_run_note(b)
    if note is not None:
        st.caption(
            note + " A high fallback rate means most of the probabilities "
            "above are calibrated from **mock** repairs, not from the model — "
            "read it before reading anything else on this page."
        )
    if b.calibration_note:
        st.info(b.calibration_note)
    if b.detection_note:
        st.info(b.detection_note)
    _detection_quality_expander(b)
    if b.comparison is not None:
        with st.expander("Calibration quality on the 30% hold-out (lower is better)"):
            comp = b.comparison
            st.table(pd.DataFrame(
                {
                    "method": ["raw", "logistic", "isotonic", "multivariate"],
                    "Brier": [comp.metric_str(m, "brier") for m in
                              ("raw", "logistic", "isotonic", "multivariate")],
                    "ECE": [comp.metric_str(m, "ece") for m in
                            ("raw", "logistic", "isotonic", "multivariate")],
                }
            ))
            st.caption(
                "'—' means the calibrator could not fit (constant labels). "
                "The lowest-Brier fitted method is the one used above."
            )


def section_costs(d: DecisionResult, c_abstain: float) -> None:
    st.header(f"5 · Expected costs ({COST_UNITS})")
    st.latex(
        r"\begin{aligned}"
        r"E[\text{AUTO\_REPAIR}] &= (1-p)\,C_{wrong} \\"
        r"E[\text{HUMAN\_REVIEW}] &= C_{human} \\"
        r"E[\text{ABSTAIN}] &= q \cdot C_{abstain}"
        r"\end{aligned}"
    )
    q = float(d.corruption_probability)
    c1, c2, c3 = st.columns(3)

    def _label(name: str, action: DecisionAction, suffix: str = "") -> str:
        base = f"E[{name}]{suffix}"
        return f"{base}  ✓ cheapest" if d.action == action else base

    c1.metric(_label("AUTO_REPAIR", DecisionAction.AUTO_REPAIR), f"{d.expected_cost_auto:.2f}")
    c2.metric(_label("HUMAN_REVIEW", DecisionAction.HUMAN_REVIEW), f"{d.expected_cost_human:.2f}")
    c3.metric(
        _label("ABSTAIN", DecisionAction.ABSTAIN, " = q × C_abstain"),
        f"{d.expected_cost_abstain:.2f}",
        help=(
            f"q × C_abstain = {q:.3f} × {c_abstain:g} = "
            f"{d.expected_cost_abstain:.2f}. Leaving a cell alone only costs "
            "the persistence penalty if the cell really was corrupted "
            "(docs/cost_semantics.md §2)."
        ),
    )
    st.caption(
        f"Abstention is charged at **q × C_abstain = {q:.3f} × "
        f"{c_abstain:g} = {d.expected_cost_abstain:.2f}** for this cell, "
        "not at the full C_abstain: a false alarm left alone leaves nothing "
        "broken behind. E[HUMAN_REVIEW] and E[ABSTAIN] are both flat in p, so "
        "the larger of the two can never be cheapest for this cell."
    )


def section_decision(d: DecisionResult) -> None:
    st.header("6 · Final decision")
    icon, kind, hint = ACTION_STYLE[d.action]
    getattr(st, kind)(f"### {icon} {d.action.value}\n{hint}")
    st.caption(d.reason)


def section_risk_demo(b: PipelineBundle, i: int, decisions: List[DecisionResult],
                      c_wrong: float, c_human: float, c_abstain: float) -> None:
    st.header("7 · Interactive risk demonstration")
    st.markdown(
        "**This is the thesis in one picture.** The calibrated probability for "
        "the selected cell is fixed; move the cost sliders in the sidebar and "
        "watch the cheapest action change. Same confidence, different "
        "consequences, different decision."
    )
    p = b.calibrated_probs[i]
    q = _q_of(b, i)
    idx = b.decision_indices()
    acts = [decisions[j].action for j in idx]

    left, right = st.columns(2)
    with left:
        st.pyplot(
            P.expected_cost_curves(c_wrong, c_human, c_abstain, p=p,
                                   corruption_probability=q),
            use_container_width=True,
        )
        st.caption(
            f"One cell, one q. The abstain line sits at q × C_abstain = "
            f"{q:.3f} × {c_abstain:g}."
        )
    with right:
        st.pyplot(
            P.decision_map(
                c_wrong, c_human, c_abstain, point=(p, q),
                points=[(b.calibrated_probs[j], _q_of(b, j), decisions[j].action)
                        for j in idx],
            ),
            use_container_width=True,
        )
        st.caption(
            "The same decision, derived over the whole (p, q) square. The star "
            "is the selected cell; the small dots are every decided cell."
        )

    counts = {a: sum(1 for x in acts if x == a) for a in DecisionAction}
    n = max(1, len(idx))
    c1, c2, c3 = st.columns(3)
    for col, action in zip((c1, c2, c3), DecisionAction):
        col.metric(f"{action.value} cells", f"{counts[action]}  ({counts[action] / n:.0%})")

    # Honest framing from docs/cost_semantics.md §4: q* is a horizontal
    # boundary, so per cell only two arms are ever live; all three actions
    # appear only across a population containing real errors and false alarms.
    q_star = P.review_abstain_threshold(c_human, c_abstain)
    n_below = sum(1 for j in idx if _q_of(b, j) < q_star)
    if q_star == float("inf"):
        where = "every decided cell (C_abstain = 0, so abstaining is free)"
    else:
        where = f"{n_below} of {len(idx)} decided cells ({n_below / n:.0%})"
    st.caption(
        f"**Which two arms are live for a given cell.** E[HUMAN_REVIEW] and "
        f"E[ABSTAIN] are both flat in p, so they never cross: abstention wins "
        f"over review exactly when q < q* = C_human/C_abstain = "
        f"{q_star:.3f}, and {where} fall below that threshold. For a single "
        f"cell, therefore, only two of the three arms are ever reachable — the "
        f"calibrated q decides *which* two, per cell, instead of the config "
        f"deciding it once for the whole dataset. All three actions occur "
        f"across the population, not within one cell "
        f"(docs/cost_semantics.md §4)."
    )

    with st.expander("All decided cells on the cost curve"):
        st.pyplot(P.decision_scatter([b.calibrated_probs[j] for j in idx], acts,
                                     c_wrong, c_human, c_abstain), use_container_width=True)


def section_baselines(b: PipelineBundle, ev: PolicyEvaluation, i: int) -> None:
    st.header("8 · Baseline comparison — evaluation against ground truth")
    st.error(
        "**Ground truth is used from here on.** Everything above was computed "
        "without the clean data (except the 0/1 labels used to fit the "
        "calibrator). This section scores the decisions after the fact.",
        icon="🔒",
    )
    if not ev.rows:
        st.warning("Nothing to evaluate.")
        return

    table = pd.DataFrame([r.to_dict() for r in ev.rows])
    table["method"] = table["method_name"].map(P._method_label)
    show = table[["method", "total_realized_cost", "mean_cost_per_record", "auto_rate",
                  "human_rate", "abstain_rate", "wrong_auto_rate", "repair_accuracy"]]
    st.dataframe(
        show.style.format({
            "total_realized_cost": "{:,.0f}", "mean_cost_per_record": "{:.3f}",
            "auto_rate": "{:.0%}", "human_rate": "{:.0%}", "abstain_rate": "{:.0%}",
            "wrong_auto_rate": "{:.0%}", "repair_accuracy": "{:.0%}",
        }),
        hide_index=True, use_container_width=True,
    )
    proposed = ev.cost_reports[ev.proposed_name]
    others = {k: v for k, v in ev.cost_reports.items() if k != ev.proposed_name}
    best_other = min(others.values(), key=lambda r: r.total_realized_cost)
    diff = best_other.total_realized_cost - proposed.total_realized_cost
    if diff > 0:
        st.success(
            f"The proposed policy realises **{diff:,.0f} {COST_UNITS} less** than the "
            f"best baseline ({P._method_label(best_other.method_name)})."
        )
    elif diff == 0:
        st.info("The proposed policy ties the best baseline on this dataset and cost setting.")
    else:
        st.warning(
            f"The proposed policy loses to {P._method_label(best_other.method_name)} by "
            f"**{-diff:,.0f} {COST_UNITS}** here. Reported as-is — not tuned away."
        )

    with st.expander("Selected cell: what actually happened (ground truth)"):
        rec = b.unc_records[i]
        clean_val = b.clean_df.reset_index(drop=True).at[rec.row_id, rec.column]
        correct = b.cal_samples[i].correct == 1
        st.markdown(
            f"Dirty `{rec.dirty_value!r}` → proposed `{rec.top_candidate!r}` · "
            f"clean value `{clean_val!r}` → repair was **{'correct' if correct else 'wrong'}**."
        )
        # The one place the corruption LABEL (ground truth) may be shown, in
        # contrast with the learned q that the decision actually used.
        if i < len(b.detection_labels):
            label = int(b.detection_labels[i])
            q = _q_of(b, i)
            method = b.detection_methods[i] if i < len(b.detection_methods) else "unknown"
            st.markdown(
                f"Was this cell **genuinely corrupted**? "
                f"**{'yes' if label else 'no — the detector raised a false alarm'}** "
                f"(ground-truth corruption label {label}). The policy did not "
                f"see this; it "
                f"used the learned q = **{q:.3f}** for the `{method}` check "
                f"type. q is a group-level rate, so a single cell can and does "
                f"disagree with it — the claim is that betting on the rate is "
                f"cheaper on average, which the table above measures."
            )


def section_plots(b: PipelineBundle, ev: PolicyEvaluation, threshold: float) -> None:
    st.header("9 · Plots")
    tabs = st.tabs(["Actions", "Realised cost", "Threshold sweep", "Confidence", "Reliability"])
    with tabs[0]:
        st.pyplot(P.action_distribution(ev.rows), use_container_width=True)
    with tabs[1]:
        st.pyplot(P.cost_comparison(ev.rows), use_container_width=True)
    with tabs[2]:
        proposed = ev.cost_reports.get(ev.proposed_name)
        st.pyplot(P.threshold_sweep(
            ev.threshold_sweep,
            proposed.total_realized_cost if proposed else None,
            threshold if threshold in ev.threshold_sweep else None,
        ), use_container_width=True)
    with tabs[3]:
        st.pyplot(P.confidence_histogram(b.raw_confidences, b.calibrated_probs,
                                         b.calibration_method), use_container_width=True)
    with tabs[4]:
        if b.comparison is not None and b.comparison.n_eval > 0:
            st.pyplot(P.reliability_figure(b.comparison), use_container_width=False)
        else:
            st.info("No calibration hold-out to plot.")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    st.set_page_config(page_title="Risk-Aware Data Repair", page_icon="🩺", layout="wide")
    st.title("Risk-Aware LLM-Assisted Data Repair")
    st.caption(
        "Calibrated probability × explicit costs → repair / review / abstain. "
        "Mock repair by default (real LLM repair is opt-in in the sidebar); "
        "all costs are experimental units."
    )

    cfg = sidebar()
    b = load_bundle(cfg["dataset"], cfg["n_records"], cfg["seed"], cfg["n_samples"],
                    cfg["iqr_k"], cfg["flag_outliers"], cfg["repair_mode"])
    costs = make_costs(cfg["dataset"], cfg["c_wrong"], cfg["c_human"], cfg["c_abstain"])
    decisions = decide_all(b, costs)

    section_overview(b)
    i = section_record(b)
    section_candidates(b, i)
    section_uncertainty(b, i)
    section_costs(decisions[i], cfg["c_abstain"])
    section_decision(decisions[i])
    section_risk_demo(b, i, decisions, cfg["c_wrong"], cfg["c_human"], cfg["c_abstain"])

    threshold = round(cfg["threshold"], 2)
    sweep = sorted({0.5, 0.6, 0.7, 0.8, 0.9, threshold})
    ev = evaluate_policies(b, costs, threshold=threshold, sweep=sweep)
    section_baselines(b, ev, i)
    section_plots(b, ev, threshold)


if __name__ == "__main__":
    main()
