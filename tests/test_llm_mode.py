"""
tests/test_llm_mode.py
----------------------
Phase 12, Step 2: ``mode="real"`` through the pipeline, the experiment
runner and the dashboard, plus per-key rate-limit rotation.

NO TEST IN THIS FILE TOUCHES THE NETWORK.
=========================================
Every test injects a transport — ``run_pipeline(llm_client=...)``, an
explicit ``UrllibTransport`` whose ``urlopen`` is faked, or a monkeypatched
``llm_repair._build_default_transport``. The autouse ``network_guard``
fixture (the same approach as ``tests/test_llm_repair.py``) replaces
``_build_default_transport``, ``urllib.request.urlopen`` and
``time.sleep``, so an accidental real transport or HTTP call fails the test
loudly instead of reaching a provider. ``TestNoNetwork`` asserts the guard
holds for the pipeline path.

NO TEST READS THE REAL `.env`.
==============================
The repository root holds a real, gitignored ``.env``. Every test here
passes an explicit :class:`LLMConfig` built from fabricated strings, and the
CLI tests monkeypatch ``run_experiment.resolve_llm_config`` so the file is
never opened. No key value is printed, asserted on, or written anywhere.

WHAT THESE TESTS ARE FOR
========================
1. **Mock mode is untouched.** ``mode="mock"`` is bit-identical to calling
   ``run_pipeline`` without a mode at all, so Phase 12 cannot silently move
   a reported number.
2. **Real mode really uses the model.** Candidates come from the fake's
   canned JSON, ``bundle.repair_mode == "real"``, ``llm_stats`` populated.
3. **Real mode degrades, never crashes.** An unconfigured key falls back for
   every cell and records it.
4. **Rotation behaves as designed**: 429 on key 1 rotates to key 2 and the
   outgoing ``Authorization`` header actually changes; an exhausted pool
   falls back cleanly.
5. **The CLI refuses honestly** and, once confirmed, writes the same
   artifact set as mock plus the repair-mode provenance.
6. **Leakage**: no clean value for any decided row reaches a request body.
"""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest

import run_experiment as rx
from src.pipeline import run_pipeline
from src.pipeline.runner import build_repairer
from src.repair import llm_repair
from src.repair.llm_repair import (
    LLMConfig,
    LLMHTTPError,
    LLMRepairer,
    LLMStats,
    UrllibTransport,
)
from src.repair.mock_repair import MockRepairer, build_repair_context

# Fabricated strings. Not credentials; they only make `is_configured` true.
FAKE_KEY = "unit-test-key-0000-not-real"
KEY_ONE = "unit-test-key-one-not-real"
KEY_TWO = "unit-test-key-two-not-real"
KEY_THREE = "unit-test-key-three-not-real"

DATA_ROOT = "datasets"
FAST_DATASET = "breast_cancer"
FAST_N = 12

# A value inside the Breast Cancer cytology range, so the canned response
# parses and casts for the numeric columns the detector flags.
CANNED_VALUE = 7


def _canned(*pairs: Any) -> str:
    """A well-formed JSON repair response."""
    pairs = pairs or ((CANNED_VALUE, 1.0),)
    return json.dumps(
        {"candidates": [{"value": v, "probability": p} for v, p in pairs],
         "reasoning": "canned test response"}
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Fakes and the network guard
# ═══════════════════════════════════════════════════════════════════════════════


class FakeClient:
    """
    Stand-in transport: records every request body, serves canned text.

    Mirrors ``tests/test_llm_repair.py``'s fake. The last queued item is
    reused once the queue runs down, which keeps batch tests short. A queued
    exception is raised instead of returned.
    """

    name = "fake"

    def __init__(self, responses: Optional[List[Any]] = None) -> None:
        self.responses: List[Any] = list(responses or [_canned()])
        self.payloads: List[Dict[str, Any]] = []
        self.calls = 0

    def chat_completion(self, payload: Dict[str, Any], timeout_s: float) -> str:
        self.calls += 1
        self.payloads.append(json.loads(json.dumps(payload)))  # deep copy
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, BaseException):
            raise item
        return str(item)

    @property
    def payload_text(self) -> str:
        return json.dumps(self.payloads)


class KeyAwareFakeClient(FakeClient):
    """
    A fake that can see which key rotation selected.

    Rotation never puts the key in the request body (leakage tests assert on
    that body), so it arrives through the optional ``set_api_key`` seam. This
    fake records the sequence of keys it was handed and 429s the ones listed
    in ``rate_limited_keys``.
    """

    def __init__(self, rate_limited_keys: List[str], responses: Optional[List[Any]] = None) -> None:
        super().__init__(responses)
        self.rate_limited_keys = list(rate_limited_keys)
        self.keys_seen: List[Optional[str]] = []
        self._key: Optional[str] = None

    def set_api_key(self, api_key: Optional[str]) -> None:
        self._key = api_key

    def chat_completion(self, payload: Dict[str, Any], timeout_s: float) -> str:
        self.keys_seen.append(self._key)
        if self._key in self.rate_limited_keys:
            self.calls += 1
            self.payloads.append(json.loads(json.dumps(payload)))
            raise LLMHTTPError(429, "rate limit exceeded for this key")
        return super().chat_completion(payload, timeout_s)


class _NetworkGuard:
    def __init__(self) -> None:
        self.transport_builds: List[str] = []
        self.http_attempts: List[str] = []
        self.sleeps: List[float] = []


@pytest.fixture(autouse=True)
def network_guard(monkeypatch) -> _NetworkGuard:
    """Any real transport construction, HTTP call or sleep fails or is recorded."""
    guard = _NetworkGuard()

    def _boom_transport(config):
        guard.transport_builds.append(repr(config))
        raise AssertionError("a real LLM transport was constructed during a test")

    def _boom_urlopen(*args, **kwargs):
        guard.http_attempts.append("urlopen")
        raise AssertionError("a real HTTP request was attempted during a test")

    monkeypatch.setattr(llm_repair, "_build_default_transport", _boom_transport)
    monkeypatch.setattr("urllib.request.urlopen", _boom_urlopen)
    monkeypatch.setattr(llm_repair.time, "sleep", guard.sleeps.append)
    return guard


def _config(**overrides: Any) -> LLMConfig:
    kwargs: Dict[str, Any] = dict(
        api_key=FAKE_KEY,
        base_url="https://fake.invalid/v1",
        model="fake-model-v1",
        temperature=0.0,
        max_tokens=256,
        timeout_s=5.0,
        max_retries=2,
        retry_backoff_s=0.0,
        cache_dir=".cache/llm-tests-never-written",
        use_cache=False,
    )
    kwargs.update(overrides)
    return LLMConfig(**kwargs)


def _real_bundle(client: Any, **kwargs: Any):
    """A small real-mode pipeline run with an injected transport."""
    opts: Dict[str, Any] = dict(
        n_records=FAST_N, data_root=DATA_ROOT, mode="real",
        llm_config=_config(), llm_client=client,
    )
    opts.update(kwargs)
    return run_pipeline(FAST_DATASET, **opts)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Mock mode is untouched
# ═══════════════════════════════════════════════════════════════════════════════


class TestMockModeUnchanged:
    def test_mode_mock_is_bit_identical_to_no_mode_argument(self):
        """
        The historical call and the explicit mock call must agree exactly —
        otherwise Phase 12 moved a reported number.
        """
        baseline = run_pipeline(FAST_DATASET, n_records=60, data_root=DATA_ROOT)
        explicit = run_pipeline(
            FAST_DATASET, n_records=60, data_root=DATA_ROOT, mode="mock"
        )

        assert baseline.raw_confidences == explicit.raw_confidences
        assert baseline.calibrated_probs == explicit.calibrated_probs
        assert baseline.corruption_probs == explicit.corruption_probs
        # repr(), not the dicts themselves: a missing-value detection carries
        # NaN as its dirty value and NaN != NaN.
        assert [repr(d.to_dict()) for d in baseline.detections] == \
               [repr(d.to_dict()) for d in explicit.detections]
        assert [repr(r.to_dict()) for r in baseline.repair_results] == \
               [repr(r.to_dict()) for r in explicit.repair_results]
        assert baseline.calibration_method == explicit.calibration_method

    def test_mock_bundle_declares_mock_and_carries_no_llm_stats(self):
        bundle = run_pipeline(FAST_DATASET, n_records=FAST_N, data_root=DATA_ROOT)

        assert bundle.repair_mode == "mock"
        assert bundle.llm_stats is None

    def test_build_repairer_mock_is_a_plain_mock_repairer(self):
        repairer = build_repairer("mock", seed=42, top_k=3)

        assert isinstance(repairer, MockRepairer)
        assert repairer.seed == 42
        assert repairer.top_k == 3

    def test_unknown_mode_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown repair mode"):
            run_pipeline(FAST_DATASET, n_records=FAST_N, data_root=DATA_ROOT, mode="llm")
        with pytest.raises(ValueError, match="Unknown repair mode"):
            build_repairer("grok")

    def test_unknown_mode_is_rejected_before_any_file_is_read(self):
        # A typo must not cost a dataset load.
        with pytest.raises(ValueError, match="Unknown repair mode"):
            run_pipeline(FAST_DATASET, data_root="no/such/folder", mode="REAL")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Real mode uses the model
# ═══════════════════════════════════════════════════════════════════════════════


class TestRealMode:
    def test_candidates_come_from_the_fake_response(self):
        client = FakeClient([_canned((CANNED_VALUE, 0.8), (3, 0.2))])

        bundle = _real_bundle(client)

        assert bundle.repair_mode == "real"
        assert client.calls == len(bundle.detections) > 0
        for result in bundle.repair_results:
            assert result.repair_method == "llm"
            assert [c.value for c in result.candidates] == [CANNED_VALUE, 3]
            assert [c.probability for c in result.candidates] == [0.8, 0.2]

    def test_llm_stats_are_populated_on_the_bundle(self):
        client = FakeClient()

        bundle = _real_bundle(client)
        stats = bundle.llm_stats

        assert stats is not None
        assert stats["cells"] == len(bundle.detections)
        assert stats["successes"] == len(bundle.detections)
        assert stats["fallbacks"] == 0
        assert stats["fallback_rate"] == 0.0
        assert stats["model"] == "fake-model-v1"
        assert stats["is_configured"] is True
        assert "rate_limited" in stats and "key_rotations" in stats
        assert json.dumps(stats)  # serialisable into run_metadata.json

    def test_no_api_key_material_in_the_bundle_stats(self):
        bundle = _real_bundle(FakeClient())

        blob = json.dumps(bundle.llm_stats)
        assert FAKE_KEY not in blob
        assert "api_key" not in bundle.llm_stats
        assert bundle.llm_stats["api_key_status"] == "<set>"

    def test_real_mode_leaves_the_decision_layer_intact(self):
        """Downstream is repairer-agnostic: p and q still exist and align."""
        bundle = _real_bundle(FakeClient())

        n = len(bundle.unc_records)
        assert len(bundle.calibrated_probs) == n
        assert len(bundle.corruption_probs) == n
        assert all(0.0 <= p <= 1.0 for p in bundle.calibrated_probs)
        assert all(0.0 <= q <= 1.0 for q in bundle.corruption_probs)

    def test_repair_results_order_matches_the_detections(self):
        bundle = _real_bundle(FakeClient())

        assert [(r.row_id, r.column) for r in bundle.repair_results] == \
               [(d.row_id, d.column) for d in bundle.detections]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Real mode without a key: fall back, never raise
# ═══════════════════════════════════════════════════════════════════════════════


class TestUnconfiguredRealMode:
    def test_every_cell_falls_back_and_it_is_recorded(self):
        bundle = run_pipeline(
            FAST_DATASET, n_records=FAST_N, data_root=DATA_ROOT, mode="real",
            llm_config=LLMConfig(api_key=None, model="unreachable-model"),
        )

        n = len(bundle.detections)
        assert n > 0
        assert bundle.repair_mode == "real"
        assert all(
            r.repair_method == "llm->mock (not_configured)"
            for r in bundle.repair_results
        )
        stats = bundle.llm_stats
        assert stats["fallbacks"] == n
        assert stats["fallback_rate"] == 1.0
        assert stats["llm_calls"] == 0
        assert stats["fallback_reasons"] == {"not_configured": n}
        assert stats["is_configured"] is False

    def test_fallback_candidates_equal_the_mock_repairers(self):
        """The fallback really is MockRepairer, so a keyless 'real' run is a
        mock run — which is exactly why the fallback rate is reported."""
        unconfigured = run_pipeline(
            FAST_DATASET, n_records=FAST_N, data_root=DATA_ROOT, mode="real",
            llm_config=LLMConfig(api_key=None),
        )
        mock = run_pipeline(FAST_DATASET, n_records=FAST_N, data_root=DATA_ROOT)

        assert [repr([c.to_dict() for c in r.candidates])
                for r in unconfigured.repair_results] == \
               [repr([c.to_dict() for c in r.candidates])
                for r in mock.repair_results]

    def test_a_placeholder_key_counts_as_unconfigured(self):
        bundle = run_pipeline(
            FAST_DATASET, n_records=FAST_N, data_root=DATA_ROOT, mode="real",
            llm_config=LLMConfig(api_key="your-key-here"),
        )

        assert bundle.llm_stats["fallback_rate"] == 1.0


class TestNoNetwork:
    def test_real_mode_with_an_injected_client_builds_no_transport(self, network_guard):
        _real_bundle(FakeClient())

        assert network_guard.transport_builds == []
        assert network_guard.http_attempts == []

    def test_unconfigured_real_mode_builds_no_transport(self, network_guard):
        run_pipeline(
            FAST_DATASET, n_records=FAST_N, data_root=DATA_ROOT, mode="real",
            llm_config=LLMConfig(api_key=None),
        )

        assert network_guard.transport_builds == []
        assert network_guard.http_attempts == []


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Key rotation
# ═══════════════════════════════════════════════════════════════════════════════


def _rotation_context():
    """A RepairContext for the first Breast Cancer detection (dirty data only)."""
    bundle = run_pipeline(FAST_DATASET, n_records=1, data_root=DATA_ROOT)
    return build_repair_context(bundle.detections[0], bundle.dirty_df, bundle.profile)


class _FakeHTTPResponse:
    """Minimal stand-in for ``urlopen``'s context manager."""

    def __init__(self, text: str) -> None:
        self._body = json.dumps(
            {"choices": [{"message": {"content": text}}]}
        ).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


class TestKeyRotationConfig:
    def test_llm_api_keys_populates_the_pool(self, monkeypatch):
        for name in ("LLM_API_KEY", "LLM_API_KEYS", "LLM_MODEL", "LLM_BASE_URL"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("LLM_API_KEYS", f" {KEY_ONE}, {KEY_TWO} ,{KEY_THREE} ")

        config = LLMConfig.from_env(dotenv_path=None)

        assert config.n_api_keys == 3
        assert config.key_pool == (KEY_ONE, KEY_TWO, KEY_THREE)
        assert config.is_configured is True
        assert config.api_key == KEY_ONE  # single-key callers still work

    def test_pool_falls_back_to_the_single_key(self, monkeypatch):
        for name in ("LLM_API_KEY", "LLM_API_KEYS"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("LLM_API_KEY", FAKE_KEY)

        config = LLMConfig.from_env(dotenv_path=None)

        assert config.key_pool == (FAKE_KEY,)
        assert config.n_api_keys == 1

    def test_placeholders_and_duplicates_are_dropped(self):
        config = LLMConfig(api_keys=(KEY_ONE, "your-key-here", KEY_ONE, "  ", KEY_TWO))

        assert config.key_pool == (KEY_ONE, KEY_TWO)
        assert config.is_configured is True

    def test_a_pool_of_only_placeholders_is_unconfigured(self):
        config = LLMConfig(api_keys=("your-key-here", "changeme"))

        assert config.key_pool == ()
        assert config.is_configured is False
        assert config.n_api_keys == 0

    def test_config_repr_and_to_dict_never_show_a_key(self):
        config = LLMConfig(api_keys=(KEY_ONE, KEY_TWO))

        assert KEY_ONE not in repr(config) and KEY_TWO not in repr(config)
        assert KEY_ONE not in json.dumps(config.to_dict())
        assert config.to_dict()["n_api_keys"] == 2


class TestKeyRotation:
    def test_429_rotates_and_the_authorization_header_changes(self, monkeypatch):
        """
        Drives the *real* ``UrllibTransport`` with a faked ``urlopen``, so the
        assertion is on the header the transport actually built — not on a
        fake's idea of one.
        """
        captured: List[Optional[str]] = []

        def fake_urlopen(request, timeout=None):
            header = request.get_header("Authorization")
            captured.append(header)
            if header == f"Bearer {KEY_ONE}":
                raise urllib.error.HTTPError(
                    request.full_url, 429, "rate limit", {}, None
                )
            return _FakeHTTPResponse(_canned())

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        config = _config(api_key=None, api_keys=(KEY_ONE, KEY_TWO))
        repairer = LLMRepairer(
            config=config, client=UrllibTransport(config), use_cache=False
        )

        result = repairer.repair(_rotation_context())

        assert result.repair_method == "llm"
        assert captured == [f"Bearer {KEY_ONE}", f"Bearer {KEY_TWO}"]
        assert captured[0] != captured[1]
        assert repairer.stats.key_rotations == 1
        assert repairer.stats.rate_limited == 1
        assert repairer.stats.retries == 0       # rotation is not a backoff retry
        assert repairer.stats.transport_attempts == 2

    def test_rotation_retries_immediately_without_sleeping(self, network_guard):
        client = KeyAwareFakeClient([KEY_ONE])
        config = _config(api_key=None, api_keys=(KEY_ONE, KEY_TWO), retry_backoff_s=5.0)
        repairer = LLMRepairer(config=config, client=client, use_cache=False)

        result = repairer.repair(_rotation_context())

        assert result.repair_method == "llm"
        assert client.keys_seen == [KEY_ONE, KEY_TWO]
        assert network_guard.sleeps == []        # cheaper than backing off
        assert repairer.stats.key_rotations == 1

    def test_a_throttled_key_is_not_retried_on_the_next_cell(self):
        """The rotated-to index persists across cells, so a key already known
        to be throttled does not cost an attempt per cell."""
        client = KeyAwareFakeClient([KEY_ONE])
        config = _config(api_key=None, api_keys=(KEY_ONE, KEY_TWO))
        repairer = LLMRepairer(config=config, client=client, use_cache=False)
        context = _rotation_context()

        first = repairer.repair(context)
        second = repairer.repair(context)

        assert first.repair_method == "llm" and second.repair_method == "llm"
        assert client.keys_seen == [KEY_ONE, KEY_TWO, KEY_TWO]
        assert repairer.stats.key_rotations == 1   # not one per cell
        assert repairer.stats.rate_limited == 1

    def test_rotation_walks_the_whole_pool(self):
        client = KeyAwareFakeClient([KEY_ONE, KEY_TWO])
        config = _config(api_key=None, api_keys=(KEY_ONE, KEY_TWO, KEY_THREE))
        repairer = LLMRepairer(config=config, client=client, use_cache=False)

        result = repairer.repair(_rotation_context())

        assert result.repair_method == "llm"
        assert client.keys_seen == [KEY_ONE, KEY_TWO, KEY_THREE]
        assert repairer.stats.key_rotations == 2
        assert repairer.stats.rate_limited == 2

    def test_all_keys_rate_limited_falls_back_cleanly(self, network_guard):
        client = KeyAwareFakeClient([KEY_ONE, KEY_TWO])
        config = _config(
            api_key=None, api_keys=(KEY_ONE, KEY_TWO), max_retries=0
        )
        repairer = LLMRepairer(config=config, client=client, use_cache=False)

        result = repairer.repair(_rotation_context())

        # No exception, a usable result, and the reason is on the record.
        assert result.repair_method == "llm->mock (request_failed)"
        assert result.candidates
        assert client.keys_seen == [KEY_ONE, KEY_TWO]
        assert repairer.stats.key_rotations == 1
        assert repairer.stats.rate_limited == 2
        assert repairer.stats.fallbacks == 1
        assert repairer.stats.fallback_reasons == {"request_failed": 1}

    def test_exhausted_pool_then_backs_off_before_giving_up(self, network_guard):
        client = KeyAwareFakeClient([KEY_ONE, KEY_TWO])
        config = _config(
            api_key=None, api_keys=(KEY_ONE, KEY_TWO),
            max_retries=2, retry_backoff_s=0.5,
        )
        repairer = LLMRepairer(config=config, client=client, use_cache=False)

        result = repairer.repair(_rotation_context())

        # Rotate once (free), then two backoff retries, then fall back.
        assert result.repair_method == "llm->mock (request_failed)"
        assert repairer.stats.key_rotations == 1
        assert repairer.stats.retries == 2
        assert network_guard.sleeps == [0.5, 1.0]

    def test_a_single_key_pool_never_rotates(self, network_guard):
        client = KeyAwareFakeClient([FAKE_KEY])
        repairer = LLMRepairer(
            config=_config(max_retries=1, retry_backoff_s=0.0),
            client=client, use_cache=False,
        )

        result = repairer.repair(_rotation_context())

        assert result.repair_method == "llm->mock (request_failed)"
        assert repairer.stats.key_rotations == 0
        assert repairer.stats.retries == 1       # backoff, since no spare key

    def test_a_5xx_does_not_rotate(self):
        """A server fault hits every key; rotating would burn the whole pool."""
        client = KeyAwareFakeClient([], responses=[LLMHTTPError(503, "unavailable")])
        config = _config(api_key=None, api_keys=(KEY_ONE, KEY_TWO), max_retries=1)
        repairer = LLMRepairer(config=config, client=client, use_cache=False)

        repairer.repair(_rotation_context())

        assert repairer.stats.key_rotations == 0
        assert repairer.stats.rate_limited == 0
        assert repairer.stats.retries == 1
        assert client.keys_seen == [KEY_ONE, KEY_ONE]

    def test_active_key_is_never_placed_in_the_request_body(self):
        client = KeyAwareFakeClient([KEY_ONE])
        config = _config(api_key=None, api_keys=(KEY_ONE, KEY_TWO))
        repairer = LLMRepairer(config=config, client=client, use_cache=False)

        repairer.repair(_rotation_context())

        for key in (KEY_ONE, KEY_TWO):
            assert key not in client.payload_text

    def test_stats_to_dict_hides_rotation_unless_asked(self):
        stats = LLMStats(cells=1)
        stats.rate_limited = 3
        stats.key_rotations = 2

        assert "key_rotations" not in stats.to_dict()
        full = stats.to_dict(include_rotation=True)
        assert full["key_rotations"] == 2
        assert full["rate_limited"] == 3

    def test_rotation_counters_reach_the_bundle(self):
        client = KeyAwareFakeClient([KEY_ONE])

        bundle = _real_bundle(
            client, llm_config=_config(api_key=None, api_keys=(KEY_ONE, KEY_TWO))
        )

        assert bundle.llm_stats["key_rotations"] == 1
        assert bundle.llm_stats["rate_limited"] == 1
        assert bundle.llm_stats["n_api_keys"] == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 5. The experiment runner
# ═══════════════════════════════════════════════════════════════════════════════


def _no_key_config() -> LLMConfig:
    return LLMConfig(api_key=None, model="fake-model-v1")


class TestCliRefusals:
    def test_mode_real_without_a_key_exits_2(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(rx, "resolve_llm_config", lambda *a, **k: _no_key_config())

        code = rx.main([
            "--mode", "real", "--llm-live", "--quick",
            "--dataset", FAST_DATASET, "--output-dir", str(tmp_path),
        ])
        captured = capsys.readouterr()
        message = captured.out + captured.err

        assert code == 2
        assert "LLM_API_KEY" in message
        assert ".env" in message
        # Nothing may be written before the refusal.
        assert list(tmp_path.rglob("*")) == []

    def test_the_refusal_prints_no_key_material(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(
            rx, "resolve_llm_config",
            lambda *a, **k: LLMConfig(api_key="your-key-here", api_keys=(FAKE_KEY,)),
        )

        rx.main([
            "--mode", "real", "--llm-live", "--dataset", FAST_DATASET,
            "--output-dir", str(tmp_path),
        ])
        message = "".join(capsys.readouterr())

        for secret in (FAKE_KEY, KEY_ONE, KEY_TWO):
            assert secret not in message

    def test_real_mode_needs_explicit_confirmation(self, monkeypatch, tmp_path, capsys):
        """A configured key is not consent to spend ~13k billed calls."""
        monkeypatch.setattr(rx, "resolve_llm_config", lambda *a, **k: _config())

        code = rx.main([
            "--mode", "real", "--quick", "--dataset", FAST_DATASET,
            "--output-dir", str(tmp_path),
        ])
        message = "".join(capsys.readouterr())

        assert code == 2
        assert "--llm-live" in message
        assert FAKE_KEY not in message
        assert list(tmp_path.rglob("*")) == []

    def test_llm_cache_flag_exists_and_defaults_on(self):
        parser = rx.build_parser()

        assert parser.parse_args(["--dataset", FAST_DATASET]).llm_cache is True
        assert parser.parse_args(
            ["--dataset", FAST_DATASET, "--no-llm-cache"]
        ).llm_cache is False
        assert parser.parse_args(
            ["--dataset", FAST_DATASET, "--llm-cache"]
        ).llm_cache is True


def _artifact_set(root: Path) -> List[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


@pytest.mark.integration
class TestCliRealRun:
    """``--quick --mode real --llm-live`` with a faked transport."""

    @pytest.fixture(scope="class")
    def real_run(self, tmp_path_factory, request) -> Path:
        out = tmp_path_factory.mktemp("real_run")
        monkeypatch = pytest.MonkeyPatch()
        request.addfinalizer(monkeypatch.undo)
        monkeypatch.setattr(
            rx, "resolve_llm_config",
            lambda *a, **k: _config(model="fake-model-v1", use_cache=False),
        )
        # The transport builder is the seam: run_experiment injects no client,
        # so this is what proves no real transport is ever constructed.
        monkeypatch.setattr(
            llm_repair, "_build_default_transport", lambda config: FakeClient()
        )
        code = rx.main([
            "--quick", "--mode", "real", "--llm-live", "--no-llm-cache",
            "--no-figures", "--dataset", FAST_DATASET, "--output-dir", str(out),
        ])
        assert code == 0
        return out

    @pytest.fixture(scope="class")
    def mock_run(self, tmp_path_factory) -> Path:
        out = tmp_path_factory.mktemp("mock_run")
        assert rx.main([
            "--quick", "--no-figures", "--dataset", FAST_DATASET,
            "--output-dir", str(out),
        ]) == 0
        return out

    def test_same_artifact_set_as_mock(self, real_run, mock_run):
        assert _artifact_set(real_run) == _artifact_set(mock_run)
        assert _artifact_set(real_run)  # and it is not empty

    def test_every_table_states_the_mode_model_and_fallback_rate(self, real_run):
        for stem in rx.TABLE_STEMS:
            text = (real_run / "tables" / f"{stem}.md").read_text(encoding="utf-8")
            assert "Repair mode: `real`" in text, stem
            assert "fake-model-v1" in text, stem
            assert "fell back to the mock repairer" in text, stem
            assert "llm_repair.py" in text, stem

    def test_tables_carry_a_repair_mode_column(self, real_run):
        for stem in rx.TABLE_STEMS:
            lines = [
                line for line in
                (real_run / "tables" / f"{stem}.csv").read_text(encoding="utf-8").splitlines()
                if not line.startswith("#")
            ]
            header = lines[0].split(",")
            assert header[-3:] == ["repair_mode", "llm_model", "llm_fallback_rate"], stem
            assert lines[1].split(",")[-3] == "real", stem

    def test_the_mock_run_still_says_mock(self, mock_run):
        text = (mock_run / "tables" / "t1_repair_quality.md").read_text(encoding="utf-8")
        assert "mock_repair.py" in text
        assert "No LLM was called" in text

    def test_metadata_records_mode_model_and_stats_without_a_key(self, real_run):
        meta = json.loads((real_run / "run_metadata.json").read_text(encoding="utf-8"))
        det = meta["deterministic"]

        assert det["mode"] == "real"
        assert det["repair_mode"] == "real"
        assert det["llm_model"] == "fake-model-v1"
        assert det["llm_cache"] is False
        stats = det["llm_stats"][FAST_DATASET]
        assert stats["cells"] > 0
        assert "key_rotations" in stats and "fallback_rate" in stats
        assert det["per_dataset"][FAST_DATASET]["repair_mode"] == "real"
        assert FAKE_KEY not in json.dumps(meta)
        # The status flag may be recorded; the value may not.
        assert "api_key" not in stats
        assert stats["api_key_status"] == "<set>"

    def test_predictions_name_the_mode_and_the_per_cell_provenance(self, real_run):
        path = real_run / "predictions" / f"{FAST_DATASET}.csv"
        text = path.read_text(encoding="utf-8")
        comments = [line for line in text.splitlines() if line.startswith("#")]
        rows = [line for line in text.splitlines() if not line.startswith("#")]

        assert any("mode=real" in c and "fake-model-v1" in c for c in comments)
        header = rows[0].split(",")
        assert "repair_method" in header
        i = header.index("repair_method")
        methods = {line.split(",")[i] for line in rows[1:]}
        assert methods
        assert all(m == "llm" or m.startswith("llm->mock") for m in methods)

    def test_the_honest_reading_states_the_fallback_count(self, real_run, capsys):
        # The closing summary is printed, not written, so re-derive it.
        text = (real_run / "tables" / "t3_risk_aware.md").read_text(encoding="utf-8")
        assert "mock repairer even in real mode" in text


# ═══════════════════════════════════════════════════════════════════════════════
# 6. The dashboard helpers
# ═══════════════════════════════════════════════════════════════════════════════


class TestDashboardHelpers:
    """
    Unit-level cover for the Phase 12 parts of ``app.py``.

    The rendered app is exercised by ``tests/test_app.py`` with AppTest; that
    suite deliberately leaves "Real LLM" unarmed, which is also what keeps it
    offline. These tests cover the three pieces the arm switch depends on,
    without a browser and without a pipeline run.
    """

    def test_load_bundle_takes_the_mode_in_its_cache_key(self):
        import inspect

        import app

        params = list(inspect.signature(app.load_bundle).parameters)
        assert "repair_mode" in params
        # The config (and therefore any key) must never be part of a cache key.
        assert not any("config" in p or "key" in p for p in params)

    def test_llm_status_reports_model_and_pool_but_no_key(self, monkeypatch):
        import app

        monkeypatch.setattr(
            app.LLMConfig, "from_env",
            classmethod(lambda cls, *a, **k: _config(api_key=None, api_keys=(KEY_ONE, KEY_TWO))),
        )

        status = app.llm_status()

        assert status == {
            "configured": True,
            "model": "fake-model-v1",
            "n_keys": 2,
            "base_url": "https://fake.invalid/v1",
        }
        assert KEY_ONE not in json.dumps(status)

    def test_llm_run_note_is_none_in_mock_mode(self):
        import app

        bundle = run_pipeline(FAST_DATASET, n_records=FAST_N, data_root=DATA_ROOT)

        assert app.llm_run_note(bundle) is None

    def test_llm_run_note_states_the_model_and_the_fallback_rate(self):
        import app

        bundle = _real_bundle(FakeClient(["not json at all"]))
        note = app.llm_run_note(bundle)

        assert note is not None
        assert "fake-model-v1" in note
        assert "fell back to mock repair (100.0%)" in note
        assert FAKE_KEY not in note


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Leakage at the pipeline level
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sentinel_dataset(tmp_path) -> Path:
    """
    A dataset whose CLEAN values are impossible to confuse with dirty ones.

    ``clean.csv``'s ``label`` column holds sentinel strings that appear
    nowhere in ``dirty.csv``, so a single occurrence of the sentinel in an
    outgoing request body is proof that ground truth reached the repairer.

    ``label`` is deliberately low-cardinality (two values over 24 rows) so
    the profiler calls it CATEGORICAL rather than ID_LIKE: an ID-like column
    is dropped from ``row_context`` entirely, which would make the sentinel
    unreachable and the test vacuous. With two values it *does* travel in the
    row context of every ``sensor`` repair — from the dirty frame if the
    leakage rule holds, from the clean frame if it does not.
    """
    root = tmp_path / "datasets" / FAST_DATASET
    root.mkdir(parents=True)
    n = 24
    clean = pd.DataFrame({
        "sensor": [float((i % 9) + 1) for i in range(n)],
        "label": ["ZZCLEANSENTINELA" if i % 2 else "ZZCLEANSENTINELB" for i in range(n)],
    })
    dirty = clean.copy()
    dirty["label"] = ["dirtylabelA" if i % 2 else "dirtylabelB" for i in range(n)]
    dirty.loc[[2, 5, 7], "sensor"] = None
    clean.to_csv(root / "clean.csv", index=False)
    dirty.to_csv(root / "dirty.csv", index=False)
    return tmp_path / "datasets"


class TestNoLeakageThroughRealMode:
    def test_no_clean_value_reaches_any_request_body(self, sentinel_dataset):
        client = FakeClient([_canned((4, 1.0))])

        bundle = run_pipeline(
            FAST_DATASET, n_records=50, data_root=str(sentinel_dataset),
            mode="real", llm_config=_config(), llm_client=client,
        )

        # Guard the guard: the sentinel must really be in the clean data, and
        # the run must really have sent something, or this proves nothing.
        assert any("ZZCLEANSENTINEL" in str(v) for v in bundle.clean_df["label"])
        assert client.calls == len(bundle.detections) > 0
        assert "ZZCLEANSENTINEL" not in client.payload_text

    def test_no_decided_rows_clean_cell_value_is_sent(self, sentinel_dataset):
        client = FakeClient([_canned((4, 1.0))])

        bundle = run_pipeline(
            FAST_DATASET, n_records=50, data_root=str(sentinel_dataset),
            mode="real", llm_config=_config(), llm_client=client,
        )

        clean = bundle.clean_df.reset_index(drop=True)
        blob = client.payload_text
        for i in bundle.decision_indices():
            record = bundle.unc_records[i]
            for column in clean.columns:
                value = clean.at[record.row_id, column]
                if isinstance(value, str):
                    assert value not in blob, (record.row_id, column)

    def test_the_payload_is_exactly_the_dirty_only_context(self, sentinel_dataset):
        """
        Structural guard: every captured body equals the body built from
        ``build_repair_context(detection, dirty_df, profile)``. The pipeline
        therefore adds nothing of its own to the request — and it has no
        other source of ground truth to add.
        """
        client = FakeClient([_canned((4, 1.0))])
        config = _config()
        bundle = run_pipeline(
            FAST_DATASET, n_records=50, data_root=str(sentinel_dataset),
            mode="real", llm_config=config, llm_client=client,
        )
        reference = LLMRepairer(config=config, client=FakeClient(), use_cache=False)

        expected = [
            reference.build_payload(
                build_repair_context(detection, bundle.dirty_df, bundle.profile)
            )
            for detection in bundle.detections
        ]
        assert client.payloads == expected

    def test_run_pipeline_has_no_clean_argument_on_the_real_path(self):
        import inspect

        params = inspect.signature(run_pipeline).parameters
        assert "mode" in params and "llm_config" in params and "llm_client" in params
        assert not any("clean" in name for name in params)
        assert not any(
            "clean" in name for name in inspect.signature(build_repairer).parameters
        )
