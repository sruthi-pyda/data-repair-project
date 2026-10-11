"""
tests/test_console.py
---------------------
src/console.py: make the report characters printable on a default
Windows console (cp1252), where box-drawing glyphs otherwise raise
UnicodeEncodeError partway through a demo's own output.

Also guards that every demo/experiment script actually calls the helper,
so a new script cannot silently reintroduce the crash.
"""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import pytest

from src.console import enable_unicode_stdout

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Characters the reports actually use.
REPORT_CHARS = "═ ─ │ → ± — ✓ ×"

SCRIPTS = [
    "demo_repair.py",
    "demo_uncertainty.py",
    "demo_calibration.py",
    "demo_risk.py",
    "experiments/smoothing_ablation.py",
    "experiments/three_actions_ablation.py",
]


class TestEnableUnicodeStdout:
    def test_reconfigures_a_cp1252_stream(self, monkeypatch):
        raw = io.BytesIO()
        stream = io.TextIOWrapper(raw, encoding="cp1252")
        monkeypatch.setattr(sys, "stdout", stream)
        monkeypatch.setattr(sys, "stderr", stream)

        with pytest.raises(UnicodeEncodeError):
            stream.write(REPORT_CHARS)
            stream.flush()

        assert enable_unicode_stdout() is True
        assert sys.stdout.encoding.lower().replace("_", "-") == "utf-8"
        sys.stdout.write(REPORT_CHARS)  # must not raise
        sys.stdout.flush()

    def test_noop_when_already_utf8(self, monkeypatch):
        stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        monkeypatch.setattr(sys, "stdout", stream)
        monkeypatch.setattr(sys, "stderr", stream)
        assert enable_unicode_stdout() is True
        assert sys.stdout.encoding.lower() == "utf-8"

    def test_idempotent(self, monkeypatch):
        stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
        monkeypatch.setattr(sys, "stdout", stream)
        monkeypatch.setattr(sys, "stderr", stream)
        assert enable_unicode_stdout() is True
        assert enable_unicode_stdout() is True
        sys.stdout.write(REPORT_CHARS)

    def test_survives_a_non_reconfigurable_stream(self, monkeypatch):
        """pytest's capture object has no .reconfigure; must not raise."""
        class Dummy:
            encoding = "cp1252"

            def write(self, _s):  # pragma: no cover - never called
                return 0

        monkeypatch.setattr(sys, "stdout", Dummy())
        monkeypatch.setattr(sys, "stderr", Dummy())
        assert enable_unicode_stdout() is False

    def test_survives_none_streams(self, monkeypatch):
        monkeypatch.setattr(sys, "stdout", None)
        monkeypatch.setattr(sys, "stderr", None)
        assert enable_unicode_stdout() is False

    def test_unencodable_character_is_replaced_not_fatal(self, monkeypatch):
        raw = io.BytesIO()
        stream = io.TextIOWrapper(raw, encoding="cp1252")
        monkeypatch.setattr(sys, "stdout", stream)
        monkeypatch.setattr(sys, "stderr", stream)
        enable_unicode_stdout()
        sys.stdout.write("═")
        sys.stdout.flush()
        assert raw.getvalue()  # something was written, nothing raised


class TestEveryScriptCallsIt:
    @pytest.mark.parametrize("script", SCRIPTS)
    def test_script_calls_enable_unicode_stdout(self, script):
        src = (PROJECT_ROOT / script).read_text(encoding="utf-8")
        assert "from src.console import enable_unicode_stdout" in src, script
        assert "enable_unicode_stdout()" in src, script


@pytest.mark.integration
class TestDemosRunOnACp1252Console:
    """The regression itself: run each demo in a subprocess whose stdout
    encoding is cp1252 and assert it does not die on its own report."""

    @pytest.mark.parametrize("script", ["demo_repair.py", "demo_uncertainty.py"])
    def test_demo_does_not_raise_unicode_error(self, script):
        import os

        # Inherit the real environment (so site-packages resolve) and
        # override only the encoding, to force the failing condition.
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "cp1252"
        proc = subprocess.run(
            [sys.executable, script], cwd=PROJECT_ROOT, env=env,
            capture_output=True, text=True, errors="replace", timeout=600,
        )
        combined = proc.stdout + proc.stderr
        assert "UnicodeEncodeError" not in combined, combined[-800:]
        assert proc.returncode == 0, combined[-800:]
