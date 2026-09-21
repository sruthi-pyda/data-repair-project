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

Works fully offline in mock mode; no API key is needed. "Real LLM" mode is
listed in the sidebar but Phase 12 is not implemented yet, so it falls back
to mock and says so.

LAYOUT
------
Sidebar   dataset · mode · three cost sliders · baseline threshold ·
          number of records · random seed
Sections  1 dataset overview
          2 one dirty record
          3 repair candidates
          4 uncertainty signals
          5 expected costs (experimental cost units)
          6 final decision
          7 interactive risk demonstration  ← the centrepiece
          8 baseline comparison  (ground truth used here, and only here)
          9 plots

LEAKAGE RULE (CLAUDE.md §3)
---------------------------
Sections 1–7 never display a clean value next to a dirty cell. The clean
value appears only in Section 8, which is visibly headed as evaluation.
The pipeline stages are in ``src/pipeline/runner.py``; ``decide_all`` has no
ground-truth parameter by construction.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

from src.decision.policy import DecisionAction, DecisionResult
from src.pipeline import (
    DATASETS,
    PipelineBundle,
    PolicyEvaluation,
    decide_all,
    evaluate_policies,
    run_pipeline,
)
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
def load_bundle(dataset: str, n_records: int, seed: int, n_samples: int) -> PipelineBundle:
    return run_pipeline(dataset, n_records=n_records, seed=seed, n_samples=n_samples)


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
    if mode == "Real LLM":
        st.sidebar.warning(
            "Phase 12 (real LLM client) is not implemented yet — falling back "
            "to mock repair. No API call is made."
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
        dataset=dataset, mode=mode, c_wrong=c_wrong, c_human=c_human,
        c_abstain=c_abstain, threshold=threshold, n_records=int(n_records),
        n_samples=int(n_samples), seed=int(seed),
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
    st.caption(f"Repair method: `{rr.repair_method}` (mock repairer — not an LLM)")
    table = pd.DataFrame(
        [{"rank": k + 1, "candidate": str(c.value), "probability": c.probability}
         for k, c in enumerate(rr.candidates)]
    )
    c1, c2 = st.columns([2, 3])
    c1.dataframe(table, hide_index=True, use_container_width=True)
    c2.pyplot(P.candidate_bar([(c.value, c.probability) for c in rr.candidates]),
              use_container_width=True)


def section_uncertainty(b: PipelineBundle, i: int) -> None:
    st.header("4 · Uncertainty signals")
    rec = b.unc_records[i]
    p_cal = b.calibrated_probs[i]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw confidence", f"{rec.raw_confidence:.3f}")
    c2.metric("Self-consistency", f"{rec.self_consistency:.2f}",
              help=f"Agreement across {rec.n_samples} resampled repair calls")
    c3.metric("Normalised entropy", f"{rec.normalized_entropy:.3f}",
              help="0 = one candidate dominates, 1 = uniform over candidates")
    c4.metric("Calibrated p(correct)", f"{p_cal:.3f}",
              delta=f"{p_cal - rec.raw_confidence:+.3f} vs raw",
              help=f"Calibrator: {b.calibration_method}")
    if b.calibration_note:
        st.info(b.calibration_note)
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


def section_costs(d: DecisionResult) -> None:
    st.header(f"5 · Expected costs ({COST_UNITS})")
    st.latex(
        r"\begin{aligned}"
        r"E[\text{AUTO\_REPAIR}] &= (1-p)\,C_{wrong} \\"
        r"E[\text{HUMAN\_REVIEW}] &= C_{human} \\"
        r"E[\text{ABSTAIN}] &= C_{abstain}"
        r"\end{aligned}"
    )
    c1, c2, c3 = st.columns(3)

    def _label(name: str, action: DecisionAction) -> str:
        return f"E[{name}]  ✓ cheapest" if d.action == action else f"E[{name}]"

    c1.metric(_label("AUTO_REPAIR", DecisionAction.AUTO_REPAIR), f"{d.expected_cost_auto:.2f}")
    c2.metric(_label("HUMAN_REVIEW", DecisionAction.HUMAN_REVIEW), f"{d.expected_cost_human:.2f}")
    c3.metric(_label("ABSTAIN", DecisionAction.ABSTAIN), f"{d.expected_cost_abstain:.2f}")


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
    st.pyplot(P.expected_cost_curves(c_wrong, c_human, c_abstain, p=p), use_container_width=True)

    idx = b.decision_indices()
    acts = [decisions[j].action for j in idx]
    counts = {a: sum(1 for x in acts if x == a) for a in DecisionAction}
    n = max(1, len(idx))
    c1, c2, c3 = st.columns(3)
    for col, action in zip((c1, c2, c3), DecisionAction):
        col.metric(f"{action.value} cells", f"{counts[action]}  ({counts[action] / n:.0%})")

    if c_human == c_abstain:
        st.caption("C_human equals C_abstain: ties go to HUMAN_REVIEW by the documented priority.")
    elif c_abstain > c_human:
        st.caption("C_abstain > C_human, so ABSTAIN can never be the cheapest action for any p.")
    else:
        st.caption("C_abstain < C_human, so HUMAN_REVIEW can never be the cheapest action for any p.")

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
        "Mock repair mode; all costs are experimental units."
    )

    cfg = sidebar()
    b = load_bundle(cfg["dataset"], cfg["n_records"], cfg["seed"], cfg["n_samples"])
    costs = make_costs(cfg["dataset"], cfg["c_wrong"], cfg["c_human"], cfg["c_abstain"])
    decisions = decide_all(b, costs)

    section_overview(b)
    i = section_record(b)
    section_candidates(b, i)
    section_uncertainty(b, i)
    section_costs(decisions[i])
    section_decision(decisions[i])
    section_risk_demo(b, i, decisions, cfg["c_wrong"], cfg["c_human"], cfg["c_abstain"])

    threshold = round(cfg["threshold"], 2)
    sweep = sorted({0.5, 0.6, 0.7, 0.8, 0.9, threshold})
    ev = evaluate_policies(b, costs, threshold=threshold, sweep=sweep)
    section_baselines(b, ev, i)
    section_plots(b, ev, threshold)


if __name__ == "__main__":
    main()
