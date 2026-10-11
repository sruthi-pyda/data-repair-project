"""
tests/test_app.py
-----------------
Phase 11: the Streamlit dashboard, driven headlessly with
``streamlit.testing.v1.AppTest``. No browser, no network, no API key.

Covers
  - the app renders every section without raising
  - the centrepiece: moving a cost slider flips the displayed decision
    while the calibrated probability for the cell stays fixed
  - switching to the Adult dataset shows the fallback note and abstains
  - "Real LLM" mode warns and falls back rather than calling anything
  - NO-LEAKAGE in the UI: no clean value is rendered before the
    evaluation section, and the evaluation section is explicitly headed
"""

from __future__ import annotations

import re

import pytest
from streamlit.testing.v1 import AppTest

APP = "app.py"
TIMEOUT = 240

SECTION_TITLES = [
    "1 · Dataset overview",
    "2 · One dirty record",
    "3 · Repair candidates",
    "4 · Uncertainty signals",
    "5 · Expected costs (experimental cost units)",
    "6 · Final decision",
    "7 · Interactive risk demonstration",
    "8 · Baseline comparison — evaluation against ground truth",
    "9 · Plots",
]


def _run() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def _decision(at: AppTest) -> str:
    """Section 6 renders one st.success/warning/info block headed '### <icon> ACTION'."""
    for kind in ("success", "warning", "info"):
        for el in getattr(at, kind):
            m = re.match(r"### \S+ (AUTO_REPAIR|HUMAN_REVIEW|ABSTAIN)", el.value)
            if m:
                return m.group(1)
    raise AssertionError("no decision block rendered")


def _metric_value(at: AppTest, label_prefix: str) -> str:
    for m in at.metric:
        if m.label.startswith(label_prefix):
            return m.value
    raise AssertionError(f"metric '{label_prefix}' not found")


@pytest.fixture(scope="module")
def app() -> AppTest:
    return _run()


class TestRenders:
    def test_all_sections_present(self, app):
        headers = [h.value for h in app.header]
        for title in SECTION_TITLES:
            assert title in headers, title

    def test_sidebar_controls(self, app):
        assert app.sidebar.selectbox(key="dataset").value == "breast_cancer"
        assert app.sidebar.radio(key="mode").value == "Mock (offline)"
        for key in ("c_wrong_breast_cancer", "c_human_breast_cancer", "c_abstain_breast_cancer",
                    "threshold", "n_records", "n_samples"):
            assert app.sidebar.slider(key=key) is not None
        assert app.sidebar.number_input(key="seed").value == 42

    def test_cost_sliders_default_to_yaml(self, app):
        assert app.sidebar.slider(key="c_wrong_breast_cancer").value == 50.0
        assert app.sidebar.slider(key="c_human_breast_cancer").value == 5.0
        assert app.sidebar.slider(key="c_abstain_breast_cancer").value == 10.0

    def test_costs_labelled_as_experimental_units(self, app):
        text = " ".join(h.value for h in app.header) + " ".join(c.value for c in app.caption)
        assert "experimental cost units" in text
        assert "$" not in text and "dollar" not in text.lower()

    def test_plots_rendered(self, app):
        # st.pyplot elements surface as image elements in AppTest.
        assert len(app.get("imgs")) >= 3


class TestInteractiveRiskDemo:
    """The whole thesis: same p, different costs, different action."""

    def test_sliders_flip_decision_with_fixed_probability(self):
        at = _run()
        p_before = _metric_value(at, "Calibrated p(correct)")
        assert _decision(at) == "HUMAN_REVIEW"

        at.sidebar.slider(key="c_wrong_breast_cancer").set_value(1.0).run()
        assert not at.exception
        assert _decision(at) == "AUTO_REPAIR"
        assert _metric_value(at, "Calibrated p(correct)") == p_before

        at.sidebar.slider(key="c_wrong_breast_cancer").set_value(50.0).run()
        at.sidebar.slider(key="c_abstain_breast_cancer").set_value(0.0).run()
        assert not at.exception
        assert _decision(at) == "ABSTAIN"
        assert _metric_value(at, "Calibrated p(correct)") == p_before

    def test_expected_cost_metrics_follow_formulas(self):
        """E[ABSTAIN] = q · C_abstain, where q is the learned corruption
        probability of the selected cell. The test re-runs the same pipeline
        the app runs (same defaults) to obtain q independently of the UI."""
        from src.pipeline import run_pipeline

        at = _run()
        bundle = run_pipeline("breast_cancer", n_records=300, seed=42, n_samples=5)
        first = bundle.decision_indices()[0]  # the app's default selection
        q = bundle.corruption_probs[first]
        p = float(_metric_value(at, "Calibrated p(correct)"))
        assert p == pytest.approx(bundle.calibrated_probs[first], abs=5e-4)
        e_auto = float(_metric_value(at, "E[AUTO_REPAIR]"))
        e_human = float(_metric_value(at, "E[HUMAN_REVIEW]"))
        e_abstain = float(_metric_value(at, "E[ABSTAIN]"))
        assert e_auto == pytest.approx((1 - p) * 50.0, abs=0.05)
        assert e_human == pytest.approx(5.0)
        assert e_abstain == pytest.approx(q * 10.0, abs=0.01)
        assert e_abstain < 10.0  # a missing-value cell: q < 1 by the prior

    def test_cheapest_action_marked_once(self):
        at = _run()
        marked = [m.label for m in at.metric if "cheapest" in m.label]
        assert len(marked) == 1
        assert _decision(at) in marked[0]

    def test_population_counts_change_with_costs(self):
        at = _run()
        auto_before = _metric_value(at, "AUTO_REPAIR cells")
        at.sidebar.slider(key="c_wrong_breast_cancer").set_value(1.0).run()
        auto_after = _metric_value(at, "AUTO_REPAIR cells")
        assert auto_before != auto_after
        assert auto_after.endswith("(100%)")


class TestDatasets:
    def test_adult_negative_result_surfaced(self):
        at = _run()
        at.sidebar.selectbox(key="dataset").set_value("adult").run()
        assert not at.exception, [str(e.value) for e in at.exception]
        infos = " ".join(i.value for i in at.info)
        assert "no calibrator" in infos.lower() or "base rate" in infos.lower()
        assert _decision(at) == "ABSTAIN"
        # Sliders reset to Adult's YAML values (original 10/2/5), not Breast Cancer's.
        assert at.sidebar.slider(key="c_abstain_adult").value == 5.0
        assert at.sidebar.slider(key="c_wrong_adult").value == 10.0

    def test_smartfactory_runs(self):
        at = _run()
        at.sidebar.selectbox(key="dataset").set_value("smartfactory").run()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert _decision(at) in ("AUTO_REPAIR", "HUMAN_REVIEW", "ABSTAIN")


class TestRealModeFallback:
    def test_real_llm_mode_warns_and_falls_back(self):
        at = _run()
        at.sidebar.radio(key="mode").set_value("Real LLM").run()
        assert not at.exception
        warnings = " ".join(w.value for w in at.sidebar.warning)
        assert "Phase 12" in warnings and "mock" in warnings.lower()
        assert _decision(at) in ("AUTO_REPAIR", "HUMAN_REVIEW", "ABSTAIN")


class TestNoLeakageInUI:
    def test_ground_truth_only_after_evaluation_header(self, app):
        """
        Walk the rendered tree in order. Before the Section 8 header no
        element may mention the clean value; the first mention must sit
        under that header.
        """
        seen_eval_header = False
        for el in app.main:
            text = getattr(el, "value", None)
            if not isinstance(text, str):
                continue
            if text.startswith("8 · Baseline comparison"):
                seen_eval_header = True
            if not seen_eval_header:
                assert "clean value" not in text.lower(), text[:120]

    def test_evaluation_section_is_explicitly_flagged(self, app):
        errors = " ".join(e.value for e in app.error)
        assert "Ground truth is used from here on" in errors


# ═══════════════════════════════════════════════════════════════════════════════
# Step 5: detector knobs, the learned q, and the (p, q) decision map
# ═══════════════════════════════════════════════════════════════════════════════

Q_LABEL = "q = P(cell genuinely corrupted)"


def _count(at: AppTest, action: str) -> int:
    """Section 7 renders '<n>  (<pct>%)' per action."""
    return int(_metric_value(at, f"{action} cells").split()[0])


def _bundle(**kw):
    """The same pipeline the app runs, for independent expectations."""
    from src.pipeline import run_pipeline

    params = dict(n_records=300, seed=42, n_samples=5)
    params.update(kw)
    return run_pipeline(params.pop("dataset", "breast_cancer"), **params)


class TestDetectorControls:
    def test_detector_knobs_exist_with_defaults(self, app):
        assert app.sidebar.slider(key="iqr_k_breast_cancer").value == 3.0
        assert app.sidebar.checkbox(key="flag_outliers_breast_cancer").value is True

    def test_detector_group_sits_above_the_cost_group(self, app):
        subheaders = [s.value for s in app.sidebar.subheader]
        assert "Detector" in subheaders
        assert subheaders.index("Detector") < subheaders.index(
            f"Costs ({app_module_cost_units()})"
        )

    def test_caption_names_the_all_three_actions_configuration(self, app):
        text = " ".join(c.value for c in app.sidebar.caption)
        assert "k=1.5" in text and "all three" in text

    def test_smartfactory_defaults_to_outlier_flagging_off(self):
        at = _run()
        at.sidebar.selectbox(key="dataset").set_value("smartfactory").run()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert at.sidebar.checkbox(key="flag_outliers_smartfactory").value is False


def app_module_cost_units() -> str:
    from src.visualization import plots as P

    return P.COST_UNITS_LABEL


class TestTighterFenceMakesAllThreeActionsLive:
    """docs/cost_semantics.md §4: all three actions occur across a population
    that contains both real errors and false alarms — breast_cancer at k=1.5."""

    def test_default_fence_has_no_abstentions_but_k_1_5_has_many(self):
        at = _run()
        assert _count(at, "ABSTAIN") == 0

        at.sidebar.slider(key="iqr_k_breast_cancer").set_value(1.5).run()
        assert not at.exception, [str(e.value) for e in at.exception]
        for action in ("AUTO_REPAIR", "HUMAN_REVIEW", "ABSTAIN"):
            assert _count(at, action) > 0, action

    def test_outlier_cell_abstains_and_flips_when_review_becomes_free(self):
        at = _run()
        at.sidebar.slider(key="iqr_k_breast_cancer").set_value(1.5).run()
        bundle = _bundle(iqr_k=1.5)
        outlier = next(j for j in bundle.decision_indices()
                       if bundle.detection_methods[j] == "iqr_outlier")

        at.selectbox(key="record_breast_cancer").set_value(outlier).run()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert _decision(at) == "ABSTAIN"
        q = bundle.corruption_probs[outlier]
        assert q < 0.1  # a near-certain false alarm
        assert float(_metric_value(at, Q_LABEL)) == pytest.approx(q, abs=5e-4)

        # q* = C_human/C_abstain = 0 makes review the live arm instead, so the
        # SAME cell with the SAME q decides differently.
        at.sidebar.slider(key="c_human_breast_cancer").set_value(0.0).run()
        assert not at.exception
        assert _decision(at) == "HUMAN_REVIEW"

    def test_missing_value_cell_does_not_abstain_at_k_1_5(self):
        at = _run()
        at.sidebar.slider(key="iqr_k_breast_cancer").set_value(1.5).run()
        bundle = _bundle(iqr_k=1.5)
        missing = next(j for j in bundle.decision_indices()
                       if bundle.detection_methods[j] == "missing_value")
        at.selectbox(key="record_breast_cancer").set_value(missing).run()
        assert _decision(at) != "ABSTAIN"
        assert bundle.corruption_probs[missing] > 0.9


class TestCorruptionProbabilityMetric:
    def test_q_metric_matches_the_bundle_for_the_selected_record(self):
        at = _run()
        bundle = _bundle()
        first = bundle.decision_indices()[0]
        shown = float(_metric_value(at, Q_LABEL))
        assert shown == pytest.approx(bundle.corruption_probs[first], abs=5e-4)
        assert shown > 0.9  # a missing_value cell: nearly every flag is real

    def test_how_q_was_learned_is_explained(self, app):
        captions = " ".join(c.value for c in app.caption)
        assert "detection_method" in captions
        assert "train split" in captions

    def test_detection_calibration_expander_labels_the_rate_as_learned(self, app):
        captions = " ".join(c.value for c in app.caption)
        assert "not a verdict on any individual cell" in captions
        assert "suspicion score" in captions

    def test_section_5_abstain_metric_is_q_times_c_abstain(self):
        at = _run()
        bundle = _bundle()
        first = bundle.decision_indices()[0]
        q = bundle.corruption_probs[first]
        e_abstain = float(_metric_value(at, "E[ABSTAIN]"))
        assert e_abstain == pytest.approx(q * 10.0, abs=0.01)
        labels = [m.label for m in at.metric if m.label.startswith("E[ABSTAIN]")]
        assert any("q × C_abstain" in lbl for lbl in labels), labels


class TestDecisionMapInSectionSeven:
    def test_threshold_caption_is_computed_not_asserted(self, app):
        """The old 'C_abstain > C_human so ABSTAIN can never win' caption is
        replaced by the q* framing of docs/cost_semantics.md §4."""
        captions = " ".join(c.value for c in app.caption)
        assert "q* = C_human/C_abstain = 0.500" in captions
        assert "can never be the cheapest action for any p" not in captions
        assert "All three actions occur across the population" in captions

    def test_both_figures_render_in_section_seven(self, app):
        # cost curves + decision map + candidate bar + the plots tab figures
        assert len(app.get("imgs")) >= 4


class TestAdultStaysAllAbstain:
    def test_every_decided_cell_abstains_via_learned_q(self):
        at = _run()
        at.sidebar.selectbox(key="dataset").set_value("adult").run()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert _decision(at) == "ABSTAIN"
        assert _count(at, "AUTO_REPAIR") == 0
        assert _count(at, "HUMAN_REVIEW") == 0
        assert _count(at, "ABSTAIN") > 0
        assert _metric_value(at, "ABSTAIN cells").endswith("(100%)")
        assert float(_metric_value(at, Q_LABEL)) < 0.05


class TestGroundTruthContrastOnlyInSectionEight:
    def test_corruption_label_is_not_rendered_before_section_eight(self, app):
        """
        Sections 1-7 may say anything about the LEARNED q (a decision input);
        only Section 8 may state this cell's 0/1 corruption label.
        """
        seen_eval_header = False
        for el in app.main:
            text = getattr(el, "value", None)
            if not isinstance(text, str):
                continue
            if text.startswith("8 · Baseline comparison"):
                seen_eval_header = True
            if not seen_eval_header:
                assert "genuinely corrupted?" not in text.lower(), text[:120]
                assert "ground-truth corruption label" not in text.lower(), text[:120]

    def test_section_eight_contrasts_the_label_with_the_learned_q(self, app):
        markdown = " ".join(m.value for m in app.markdown)
        assert "Was this cell **genuinely corrupted**?" in markdown
        assert "ground-truth corruption label" in markdown
        assert "used the learned q" in markdown
