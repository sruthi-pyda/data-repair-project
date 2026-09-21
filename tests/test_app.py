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
        at = _run()
        p = float(_metric_value(at, "Calibrated p(correct)"))
        e_auto = float(_metric_value(at, "E[AUTO_REPAIR]"))
        e_human = float(_metric_value(at, "E[HUMAN_REVIEW]"))
        e_abstain = float(_metric_value(at, "E[ABSTAIN]"))
        assert e_auto == pytest.approx((1 - p) * 50.0, abs=0.05)
        assert e_human == pytest.approx(5.0)
        assert e_abstain == pytest.approx(10.0)

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
        # Sliders reset to Adult's YAML values, not Breast Cancer's.
        assert at.sidebar.slider(key="c_abstain_adult").value == 1.0

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
