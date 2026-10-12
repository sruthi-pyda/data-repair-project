"""
tests/test_llm_repair.py
-------------------------
Behavioural tests for Phase 12 Step 1: src/repair/llm_repair.py.

NO TEST IN THIS FILE TOUCHES THE NETWORK.
=========================================
Every test injects a `FakeClient` through `LLMRepairer(client=...)`. An autouse
fixture additionally replaces `llm_repair._build_default_transport` and
`urllib.request.urlopen` with functions that raise, so an accidental real
transport construction or HTTP call fails the test loudly rather than reaching
a provider. `TestNoNetwork` asserts that guard actually holds.

NO REAL API KEY APPEARS HERE.
=============================
`FAKE_KEY` is a fabricated string. `TestKeyRedaction` asserts it never reaches
`repr()`, `str()`, `to_dict()` or a log record.

LEAKAGE (CLAUDE.md §3)
======================
`TestLeakage` builds a tiny clean/dirty pair whose clean values are deliberately
disjoint from the dirty ones, passes only the dirty context to the repairer, and
asserts that the *captured request body* contains none of the clean values and
no other row's data.
"""

from __future__ import annotations

import copy
import inspect
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest

from src.data.schema import (
    CategoricalStats,
    ColumnKind,
    ColumnProfile,
    DatasetProfile,
    NumericalStats,
)
from src.detection.detector import DetectionRecord, ErrorType
from src.repair import llm_repair
from src.repair.llm_repair import (
    DEFAULT_BASE_URL,
    DiskCache,
    FallbackReason,
    LLMConfig,
    LLMHTTPError,
    LLMRepairer,
    LLMStats,
    LLMTimeoutError,
    LLMTransportError,
    is_retryable_status,
)
from src.repair.mock_repair import (
    MockRepairer,
    RepairCandidate,
    RepairContext,
    RepairResult,
    build_repair_context,
)
from src.repair.prompts import (
    PARSE_EMPTY_CANDIDATES,
    PARSE_INVALID_PROBABILITIES,
    PARSE_MALFORMED_RESPONSE,
    parse_llm_response,
    parse_llm_response_detailed,
)

# A fabricated key. Not a credential; used only to make `is_configured` true.
FAKE_KEY = "unit-test-key-0000-not-real"


# ═══════════════════════════════════════════════════════════════════════════════
# Fake transport + network guard
# ═══════════════════════════════════════════════════════════════════════════════


class FakeClient:
    """
    Stand-in for `LLMTransport`.

    Records every request body it is handed (deep-copied, so later mutation
    cannot hide a leak) and serves queued responses. A queued item that is an
    exception is raised instead of returned. When the queue has a single item
    left it is reused, which keeps batch tests short.
    """

    name = "fake"

    def __init__(self, responses: Optional[List[Any]] = None) -> None:
        self.responses: List[Any] = list(responses or [])
        self.payloads: List[Dict[str, Any]] = []
        self.timeouts: List[float] = []
        self.calls = 0

    def chat_completion(self, payload: Dict[str, Any], timeout_s: float) -> str:
        self.calls += 1
        self.payloads.append(copy.deepcopy(payload))
        self.timeouts.append(timeout_s)

        if not self.responses:
            raise AssertionError("FakeClient ran out of queued responses")
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]

        if isinstance(item, BaseException):
            raise item
        if isinstance(item, type) and issubclass(item, BaseException):
            raise item("queued failure")
        return str(item)

    @property
    def payload_text(self) -> str:
        """Every captured request body as one JSON string, for leak scanning."""
        return json.dumps(self.payloads)


class _NetworkGuard:
    def __init__(self) -> None:
        self.transport_builds: List[str] = []
        self.http_attempts: List[str] = []
        self.sleeps: List[float] = []


@pytest.fixture(autouse=True)
def network_guard(monkeypatch) -> _NetworkGuard:
    """
    Make any real transport construction, HTTP call or sleep fail/record.

    Autouse: it protects every test in this module, including ones that forget
    to inject a client.
    """
    guard = _NetworkGuard()

    def _boom_transport(config):
        guard.transport_builds.append(repr(config))
        raise AssertionError("a real LLM transport was constructed during a test")

    def _boom_urlopen(*args, **kwargs):
        guard.http_attempts.append("urlopen")
        raise AssertionError("a real HTTP request was attempted during a test")

    def _record_sleep(seconds: float) -> None:
        guard.sleeps.append(seconds)

    monkeypatch.setattr(llm_repair, "_build_default_transport", _boom_transport)
    monkeypatch.setattr("urllib.request.urlopen", _boom_urlopen)
    monkeypatch.setattr(llm_repair.time, "sleep", _record_sleep)
    return guard


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures: contexts, profiles, configs
# ═══════════════════════════════════════════════════════════════════════════════


def _num_stats() -> NumericalStats:
    return NumericalStats(
        mean=5.0, std=2.0, min=1.0, q25=3.0, q50=5.0, q75=7.0, max=10.0, iqr=4.0
    )


def _int_context(**overrides: Any) -> RepairContext:
    """RepairContext for an int64 numerical column with an out-of-range value."""
    base: Dict[str, Any] = dict(
        dataset="test_ds",
        column="Bare Nuclei",
        kind=ColumnKind.NUMERICAL,
        dtype="int64",
        dirty_value=33,
        error_type=ErrorType.OUT_OF_RANGE.value,
        row_id=7,
        row_context={"Clump Thickness": 5, "Cell Size": 3},
        vocabulary=[],
        column_stats={
            "min": 1.0, "max": 10.0, "median": 3.0,
            "q25": 1.0, "q75": 5.0, "iqr": 4.0,
        },
    )
    base.update(overrides)
    return RepairContext(**base)


def _cat_context(**overrides: Any) -> RepairContext:
    """RepairContext for a categorical object column with a novel category."""
    base: Dict[str, Any] = dict(
        dataset="test_ds",
        column="occupation",
        kind=ColumnKind.CATEGORICAL,
        dtype="object",
        dirty_value="Enginer",
        error_type=ErrorType.NOVEL_CATEGORY.value,
        row_id=2,
        row_context={"age": 31},
        vocabulary=["Engineer", "Doctor", "Teacher"],
        column_stats=None,
    )
    base.update(overrides)
    return RepairContext(**base)


def _config(tmp_path: Optional[Path] = None, **overrides: Any) -> LLMConfig:
    kwargs: Dict[str, Any] = dict(
        api_key=FAKE_KEY,
        base_url="https://fake.invalid/v1",
        model="test-model",
        temperature=0.0,
        max_tokens=256,
        timeout_s=12.5,
        max_retries=2,
        retry_backoff_s=0.0,
        cache_dir=str(tmp_path / "llmcache") if tmp_path else ".cache/llm-tests",
    )
    kwargs.update(overrides)
    return LLMConfig(**kwargs)


def _repairer(
    client: Optional[FakeClient],
    tmp_path: Optional[Path] = None,
    use_cache: bool = False,
    config: Optional[LLMConfig] = None,
    **kwargs: Any,
) -> LLMRepairer:
    """Build an LLMRepairer with an injected client and caching off by default."""
    return LLMRepairer(
        config=config if config is not None else _config(tmp_path),
        client=client,
        use_cache=use_cache,
        **kwargs,
    )


def _resp(*pairs: Any, reasoning: str = "because") -> str:
    """Build a well-formed JSON response from (value, probability) pairs."""
    return json.dumps(
        {
            "candidates": [{"value": v, "probability": p} for v, p in pairs],
            "reasoning": reasoning,
        }
    )


# ═══════════════════════════════════════════════════════════════════════════════
# LLMConfig
# ═══════════════════════════════════════════════════════════════════════════════


class TestLLMConfigFromEnv:
    def _clear_env(self, monkeypatch) -> None:
        for name in (
            "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "LLM_TEMPERATURE",
            "LLM_MAX_TOKENS", "LLM_TIMEOUT_S", "LLM_MAX_RETRIES",
            "LLM_RETRY_BACKOFF_S", "LLM_CACHE_DIR",
        ):
            monkeypatch.delenv(name, raising=False)

    def test_from_env_with_key_is_configured(self, monkeypatch):
        self._clear_env(monkeypatch)
        monkeypatch.setenv("LLM_API_KEY", FAKE_KEY)
        monkeypatch.setenv("LLM_MODEL", "grok-test")
        monkeypatch.setenv("LLM_BASE_URL", "https://api.x.ai/v1")

        cfg = LLMConfig.from_env(dotenv_path=None)

        assert cfg.is_configured is True
        assert cfg.model == "grok-test"
        assert cfg.base_url == "https://api.x.ai/v1"
        assert cfg.effective_base_url == "https://api.x.ai/v1"

    def test_from_env_without_key_is_not_configured(self, monkeypatch):
        self._clear_env(monkeypatch)

        cfg = LLMConfig.from_env(dotenv_path=None)

        assert cfg.api_key is None
        assert cfg.is_configured is False
        assert cfg.api_key_status == "<unset>"
        assert cfg.effective_base_url == DEFAULT_BASE_URL

    @pytest.mark.parametrize(
        "placeholder",
        ["your-key-here", "YOUR-KEY-HERE", "your_api_key", "changeme", "   ", "sk-..."],
    )
    def test_placeholder_key_counts_as_unconfigured(self, monkeypatch, placeholder):
        self._clear_env(monkeypatch)
        monkeypatch.setenv("LLM_API_KEY", placeholder)

        cfg = LLMConfig.from_env(dotenv_path=None)

        assert cfg.is_configured is False

    def test_from_env_reads_dotenv_file(self, monkeypatch, tmp_path):
        self._clear_env(monkeypatch)
        env_file = tmp_path / ".env"
        env_file.write_text(
            "# comment line\n"
            f"LLM_API_KEY={FAKE_KEY}\n"
            "LLM_MODEL='dotenv-model'\n"
            "LLM_BASE_URL=https://api.x.ai/v1\n",
            encoding="utf-8",
        )

        cfg = LLMConfig.from_env(dotenv_path=env_file)

        assert cfg.is_configured is True
        assert cfg.model == "dotenv-model"
        assert cfg.base_url == "https://api.x.ai/v1"

    def test_real_environment_overrides_dotenv(self, monkeypatch, tmp_path):
        self._clear_env(monkeypatch)
        env_file = tmp_path / ".env"
        env_file.write_text("LLM_MODEL=from-file\n", encoding="utf-8")
        monkeypatch.setenv("LLM_MODEL", "from-environ")

        cfg = LLMConfig.from_env(dotenv_path=env_file)

        assert cfg.model == "from-environ"

    def test_missing_dotenv_file_is_not_an_error(self, monkeypatch, tmp_path):
        self._clear_env(monkeypatch)

        cfg = LLMConfig.from_env(dotenv_path=tmp_path / "nope.env")

        assert cfg.is_configured is False

    def test_numeric_env_values_parsed_and_bad_values_defaulted(self, monkeypatch):
        self._clear_env(monkeypatch)
        monkeypatch.setenv("LLM_TEMPERATURE", "0.7")
        monkeypatch.setenv("LLM_MAX_TOKENS", "128")
        monkeypatch.setenv("LLM_MAX_RETRIES", "banana")

        cfg = LLMConfig.from_env(dotenv_path=None)

        assert cfg.temperature == pytest.approx(0.7)
        assert cfg.max_tokens == 128
        assert cfg.max_retries == 2  # default survives a junk value

    def test_defaults_are_deterministic(self):
        cfg = LLMConfig()
        assert cfg.temperature == 0.0
        assert cfg.max_retries == 2
        assert cfg.cache_dir == ".cache/llm"


class TestKeyRedaction:
    def test_repr_and_str_do_not_leak_the_key(self):
        cfg = _config(api_key=FAKE_KEY)

        assert FAKE_KEY not in repr(cfg)
        assert FAKE_KEY not in str(cfg)
        assert "api_key='<set>'" in repr(cfg)

    def test_repr_marks_an_unset_key(self):
        cfg = LLMConfig(api_key=None)
        assert "api_key='<unset>'" in repr(cfg)

    def test_to_dict_exposes_status_not_value(self):
        cfg = _config(api_key=FAKE_KEY)

        payload = json.dumps(cfg.to_dict())

        assert FAKE_KEY not in payload
        assert cfg.to_dict()["api_key_status"] == "<set>"
        assert "api_key" not in cfg.to_dict()

    def test_repairer_logs_do_not_leak_the_key(self, caplog):
        # A provider that echoes the Authorization header back in an error body
        # must still not put the key in our logs.
        client = FakeClient([LLMHTTPError(401, f"invalid key {FAKE_KEY}")])
        repairer = _repairer(client)

        with caplog.at_level(logging.DEBUG, logger="src.repair.llm_repair"):
            result = repairer.repair(_int_context())

        assert result.repair_method == "llm->mock (request_failed)"
        assert FAKE_KEY not in caplog.text
        assert "<redacted>" in caplog.text

    def test_redact_helper_handles_none_and_short_secrets(self):
        assert llm_repair._redact(None, FAKE_KEY) == ""
        assert llm_repair._redact("abc", None) == "abc"
        assert llm_repair._redact("abc", "ab") == "abc"  # too short to redact


# ═══════════════════════════════════════════════════════════════════════════════
# No network
# ═══════════════════════════════════════════════════════════════════════════════


class TestNoNetwork:
    def test_injected_client_means_no_transport_is_built(self, network_guard):
        client = FakeClient([_resp((3, 1.0))])
        repairer = _repairer(client)

        repairer.repair(_int_context())

        assert client.calls == 1
        assert network_guard.transport_builds == []
        assert network_guard.http_attempts == []

    def test_unconfigured_repairer_never_builds_a_transport(self, network_guard):
        repairer = LLMRepairer(config=LLMConfig(api_key=None), use_cache=False)

        result = repairer.repair(_int_context())

        assert result.repair_method == "llm->mock (not_configured)"
        assert repairer._transport is None
        assert network_guard.transport_builds == []

    def test_building_a_repairer_opens_nothing(self, network_guard):
        LLMRepairer(config=_config(), client=FakeClient(), use_cache=False)
        assert network_guard.transport_builds == []
        assert network_guard.http_attempts == []


# ═══════════════════════════════════════════════════════════════════════════════
# Happy path
# ═══════════════════════════════════════════════════════════════════════════════


class TestHappyPath:
    def test_candidates_returned_in_probability_order(self):
        client = FakeClient([_resp((3, 0.2), (5, 0.7), (4, 0.1))])
        repairer = _repairer(client)

        result = repairer.repair(_int_context())

        probs = [c.probability for c in result.candidates]
        assert probs == sorted(probs, reverse=True)
        assert result.candidates[0].value == 5
        assert result.top_candidate is result.candidates[0]
        assert math.isclose(sum(probs), 1.0, abs_tol=1e-6)

    def test_repair_method_is_llm_on_success(self):
        client = FakeClient([_resp((3, 1.0))])
        repairer = _repairer(client)

        result = repairer.repair(_int_context())

        assert result.repair_method == "llm"
        assert repairer.stats.successes == 1
        assert repairer.stats.fallbacks == 0
        assert repairer.stats.llm_calls == 1
        assert repairer.stats.cells == 1

    def test_result_type_and_fields_mirror_the_context(self):
        client = FakeClient([_resp((3, 1.0))])
        ctx = _int_context()

        result = _repairer(client).repair(ctx)

        assert isinstance(result, RepairResult)
        assert result.dataset == ctx.dataset
        assert result.row_id == ctx.row_id
        assert result.column == ctx.column
        assert result.dirty_value == ctx.dirty_value
        assert result.error_type == ctx.error_type
        assert all(isinstance(c, RepairCandidate) for c in result.candidates)
        assert result.to_dict()["repair_method"] == "llm"
        assert result.context_used["model"] == "test-model"
        assert result.context_used["from_cache"] is False

    def test_json_in_a_code_fence_is_accepted(self):
        client = FakeClient(["Sure!\n```json\n" + _resp((3, 1.0)) + "\n```"])

        result = _repairer(client).repair(_int_context())

        assert result.repair_method == "llm"
        assert result.candidates[0].value == 3

    def test_payload_is_an_openai_chat_completions_body(self):
        client = FakeClient([_resp((3, 1.0))])
        repairer = _repairer(client)

        repairer.repair(_int_context())
        payload = client.payloads[0]

        assert payload["model"] == "test-model"
        assert payload["temperature"] == 0.0
        assert payload["max_tokens"] == 256
        assert [m["role"] for m in payload["messages"]] == ["system", "user"]
        assert "Bare Nuclei" in payload["messages"][1]["content"]
        assert client.timeouts == [12.5]

    def test_top_k_is_passed_through_to_the_prompt(self):
        client = FakeClient([_resp((3, 1.0))])
        repairer = _repairer(client, top_k=2)

        repairer.repair(_int_context())

        system = client.payloads[0]["messages"][0]["content"]
        user = client.payloads[0]["messages"][1]["content"]
        assert "up to 2 repair candidates" in system
        assert "up to 2 repair candidates" in user

    def test_categorical_column_keeps_string_values(self):
        client = FakeClient([_resp(("Engineer", 0.8), ("Doctor", 0.2))])

        result = _repairer(client).repair(_cat_context())

        assert result.candidates[0].value == "Engineer"
        assert isinstance(result.candidates[0].value, str)


# ═══════════════════════════════════════════════════════════════════════════════
# Fallbacks — one test per FallbackReason
# ═══════════════════════════════════════════════════════════════════════════════


class TestFallbackReasons:
    def _assert_fell_back(self, result: RepairResult, repairer: LLMRepairer,
                          reason: FallbackReason) -> None:
        assert result.repair_method == f"llm->mock ({reason.value})"
        assert repairer.stats.fallbacks == 1
        assert repairer.stats.fallback_reasons == {reason.value: 1}
        assert repairer.stats.successes == 0
        # A fallback still produces a usable repair.
        assert result.candidates
        assert result.context_used["fallback_reason"] == reason.value
        assert "mock_method" in result.context_used

    def test_not_configured(self):
        repairer = LLMRepairer(
            config=_config(api_key=None), client=FakeClient(), use_cache=False
        )

        result = repairer.repair(_int_context())

        self._assert_fell_back(result, repairer, FallbackReason.NOT_CONFIGURED)
        assert repairer.stats.llm_calls == 0

    def test_request_failed_on_permanent_http_error(self):
        client = FakeClient([LLMHTTPError(400, "bad request")])
        repairer = _repairer(client)

        result = repairer.repair(_int_context())

        self._assert_fell_back(result, repairer, FallbackReason.REQUEST_FAILED)
        assert client.calls == 1

    def test_request_failed_on_transport_error(self):
        client = FakeClient([LLMTransportError("connection refused")])
        repairer = _repairer(client)

        result = repairer.repair(_int_context())

        self._assert_fell_back(result, repairer, FallbackReason.REQUEST_FAILED)

    def test_request_failed_on_unexpected_exception(self):
        # repair() must never raise, whatever a transport does.
        client = FakeClient([ValueError("something nobody predicted")])
        repairer = _repairer(client)

        result = repairer.repair(_int_context())

        self._assert_fell_back(result, repairer, FallbackReason.REQUEST_FAILED)

    def test_timeout(self):
        client = FakeClient([LLMTimeoutError("timed out after 30s")])
        repairer = _repairer(client)

        result = repairer.repair(_int_context())

        self._assert_fell_back(result, repairer, FallbackReason.TIMEOUT)
        assert client.calls == 1  # timeouts are not retried

    def test_malformed_response_non_json(self):
        client = FakeClient(["I'm sorry, I cannot help with that."])
        repairer = _repairer(client)

        result = repairer.repair(_int_context())

        self._assert_fell_back(result, repairer, FallbackReason.MALFORMED_RESPONSE)

    def test_malformed_response_missing_candidates_key(self):
        client = FakeClient(['{"reasoning": "no candidates key here"}'])
        repairer = _repairer(client)

        result = repairer.repair(_int_context())

        self._assert_fell_back(result, repairer, FallbackReason.MALFORMED_RESPONSE)

    def test_malformed_response_uncastable_value(self):
        client = FakeClient([_resp(("not-a-number", 1.0))])  # int64 column
        repairer = _repairer(client)

        result = repairer.repair(_int_context())

        self._assert_fell_back(result, repairer, FallbackReason.MALFORMED_RESPONSE)

    def test_malformed_response_null_value(self):
        client = FakeClient([_resp((None, 1.0))])
        repairer = _repairer(client)

        result = repairer.repair(_int_context())

        self._assert_fell_back(result, repairer, FallbackReason.MALFORMED_RESPONSE)

    def test_empty_candidates(self):
        client = FakeClient(['{"candidates": [], "reasoning": "nothing"}'])
        repairer = _repairer(client)

        result = repairer.repair(_int_context())

        self._assert_fell_back(result, repairer, FallbackReason.EMPTY_CANDIDATES)

    @pytest.mark.parametrize(
        "body",
        [
            '{"candidates": [{"value": 3, "probability": -0.5}]}',     # negative
            '{"candidates": [{"value": 3, "probability": 1.5}]}',      # > 1
            '{"candidates": [{"value": 3, "probability": NaN}]}',      # NaN
            '{"candidates": [{"value": 3, "probability": 0}]}',        # zero
            '{"candidates": [{"value": 3, "probability": "high"}]}',   # non-numeric
            '{"candidates": [{"value": 3, "probability": 1.0}, '
            '{"value": 5, "probability": 1.0}]}',                      # sums to 2.0
        ],
    )
    def test_invalid_probabilities(self, body):
        client = FakeClient([body])
        repairer = _repairer(client)

        result = repairer.repair(_int_context())

        self._assert_fell_back(result, repairer, FallbackReason.INVALID_PROBABILITIES)

    def test_every_fallback_reason_is_covered_by_the_enum(self):
        # Guards against a new reason being added without a test above.
        assert {r.value for r in FallbackReason} == {
            "not_configured", "request_failed", "malformed_response",
            "invalid_probabilities", "empty_candidates", "timeout",
        }

    def test_fallback_matches_the_mock_repairer_candidates(self):
        ctx = _int_context()
        client = FakeClient(["garbage"])
        repairer = _repairer(client, seed=42, top_k=3)

        fallback_result = repairer.repair(ctx)
        mock_result = MockRepairer(seed=42, top_k=3).repair(ctx)

        assert [c.to_dict() for c in fallback_result.candidates] == [
            c.to_dict() for c in mock_result.candidates
        ]
        assert fallback_result.context_used["mock_method"] == mock_result.repair_method

    def test_arrow_in_repair_method_is_ascii(self):
        client = FakeClient(["garbage"])

        result = _repairer(client).repair(_int_context())

        assert "->" in result.repair_method
        assert result.repair_method.isascii()


# ═══════════════════════════════════════════════════════════════════════════════
# Validation that repairs rather than rejects
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidationRepairs:
    @pytest.mark.parametrize("scale", [0.9, 1.1])
    def test_probabilities_renormalised(self, scale):
        body = _resp((3, 0.6 * scale), (5, 0.4 * scale))
        client = FakeClient([body])

        result = _repairer(client).repair(_int_context())

        assert result.repair_method == "llm"
        total = sum(c.probability for c in result.candidates)
        assert math.isclose(total, 1.0, abs_tol=1e-6)
        assert result.candidates[0].probability == pytest.approx(0.6, abs=1e-3)

    def test_duplicate_values_collapsed_with_summed_probability(self):
        client = FakeClient([_resp((3, 0.4), (3, 0.3), (5, 0.3))])

        result = _repairer(client).repair(_int_context())

        assert result.repair_method == "llm"
        assert [c.value for c in result.candidates] == [3, 5]
        assert result.candidates[0].probability == pytest.approx(0.7, abs=1e-6)

    def test_duplicates_across_types_collapse_after_casting(self):
        # int64 column: 3, "3" and 3.0 are one candidate, not three.
        client = FakeClient([_resp((3, 0.4), ("3", 0.3), (3.0, 0.3))])

        result = _repairer(client).repair(_int_context())

        assert len(result.candidates) == 1
        assert result.candidates[0].value == 3
        assert result.candidates[0].probability == pytest.approx(1.0)

    def test_more_than_top_k_is_truncated_and_renormalised(self):
        client = FakeClient(
            [_resp((1, 0.4), (2, 0.3), (3, 0.15), (4, 0.1), (5, 0.05))]
        )
        repairer = _repairer(client, top_k=3)

        result = repairer.repair(_int_context())

        assert result.repair_method == "llm"
        assert len(result.candidates) == 3
        assert [c.value for c in result.candidates] == [1, 2, 3]
        assert math.isclose(sum(c.probability for c in result.candidates), 1.0, abs_tol=1e-6)

    def test_int_column_casts_to_int(self):
        client = FakeClient([_resp(("3", 0.6), (4.4, 0.4))])

        result = _repairer(client).repair(_int_context())

        assert result.candidates[0].value == 3
        assert isinstance(result.candidates[0].value, int)
        assert result.candidates[1].value == 4  # 4.4 rounds to 4
        assert isinstance(result.candidates[1].value, int)

    def test_float_column_casts_to_float(self):
        ctx = _int_context(dtype="float64")
        client = FakeClient([_resp(("3.5", 1.0))])

        result = _repairer(client).repair(ctx)

        assert result.candidates[0].value == pytest.approx(3.5)
        assert isinstance(result.candidates[0].value, float)

    def test_sum_tolerance_boundary_rejects_a_wild_sum(self):
        # 0.7 is outside the 0.25 tolerance → rejected, not renormalised.
        client = FakeClient([_resp((3, 0.4), (5, 0.3))])
        repairer = _repairer(client)

        result = repairer.repair(_int_context())

        assert result.repair_method == "llm->mock (invalid_probabilities)"


# ═══════════════════════════════════════════════════════════════════════════════
# Cache
# ═══════════════════════════════════════════════════════════════════════════════


class TestCache:
    def test_cache_hit_avoids_a_second_transport_call(self, tmp_path):
        client = FakeClient([_resp((3, 0.7), (5, 0.3))])
        repairer = _repairer(client, tmp_path, use_cache=True)
        ctx = _int_context()

        first = repairer.repair(ctx)
        second = repairer.repair(ctx)

        assert client.calls == 1
        assert repairer.stats.cache_hits == 1
        assert repairer.stats.llm_calls == 1
        assert first.repair_method == second.repair_method == "llm"
        assert [c.to_dict() for c in first.candidates] == [
            c.to_dict() for c in second.candidates
        ]
        assert second.context_used["from_cache"] is True

    def test_cache_writes_a_json_file(self, tmp_path):
        client = FakeClient([_resp((3, 1.0))])
        repairer = _repairer(client, tmp_path, use_cache=True)

        repairer.repair(_int_context())

        files = list((tmp_path / "llmcache").glob("*.json"))
        assert len(files) == 1
        payload = json.loads(files[0].read_text(encoding="utf-8"))
        assert payload["model"] == "test-model"
        assert "candidates" in payload["response"]
        assert FAKE_KEY not in files[0].read_text(encoding="utf-8")

    def test_use_cache_false_calls_twice(self, tmp_path):
        client = FakeClient([_resp((3, 1.0))])
        repairer = _repairer(client, tmp_path, use_cache=False)
        ctx = _int_context()

        repairer.repair(ctx)
        repairer.repair(ctx)

        assert client.calls == 2
        assert repairer.stats.cache_hits == 0
        assert not (tmp_path / "llmcache").exists()

    def test_cache_key_depends_on_model_and_prompt(self):
        k1 = DiskCache.make_key("model-a", "prompt")
        k2 = DiskCache.make_key("model-b", "prompt")
        k3 = DiskCache.make_key("model-a", "other prompt")

        assert k1 != k2 != k3
        assert k1 == DiskCache.make_key("model-a", "prompt")
        assert len(k1) == 64  # sha256 hex

    def test_different_cells_do_not_share_a_cache_entry(self, tmp_path):
        client = FakeClient([_resp((3, 1.0))])
        repairer = _repairer(client, tmp_path, use_cache=True)

        repairer.repair(_int_context(row_id=1))
        repairer.repair(_int_context(row_id=2))

        assert client.calls == 2
        assert repairer.stats.cache_hits == 0

    def test_corrupt_cache_entry_is_treated_as_a_miss(self, tmp_path):
        cache = DiskCache(tmp_path / "llmcache")
        key = DiskCache.make_key("test-model", "whatever")
        cache.directory.mkdir(parents=True, exist_ok=True)
        cache.path_for(key).write_text("{not json", encoding="utf-8")

        assert cache.get(key) is None
        assert cache.get("never-written") is None

    def test_an_injected_cache_object_is_used(self, tmp_path):
        class MemoryCache:
            def __init__(self):
                self.store: Dict[str, str] = {}
                self.gets = 0

            def make_key(self, model, prompt):
                return f"{model}|{len(prompt)}"

            def get(self, key):
                self.gets += 1
                return self.store.get(key)

            def set(self, key, response, model=""):
                self.store[key] = response

        cache = MemoryCache()
        client = FakeClient([_resp((3, 1.0))])
        repairer = LLMRepairer(
            config=_config(tmp_path), client=client, cache=cache, use_cache=True
        )
        ctx = _int_context()

        repairer.repair(ctx)
        repairer.repair(ctx)

        assert client.calls == 1
        assert cache.gets == 2
        assert len(cache.store) == 1

    def test_cache_path_argument_builds_a_disk_cache(self, tmp_path):
        repairer = LLMRepairer(
            config=_config(tmp_path), client=FakeClient(), cache=tmp_path / "elsewhere"
        )
        assert isinstance(repairer.cache, DiskCache)
        assert repairer.cache.directory == tmp_path / "elsewhere"


# ═══════════════════════════════════════════════════════════════════════════════
# Retries
# ═══════════════════════════════════════════════════════════════════════════════


class TestRetries:
    def test_retry_on_429_then_success(self):
        client = FakeClient([LLMHTTPError(429, "slow down"), _resp((3, 1.0))])
        repairer = _repairer(client)

        result = repairer.repair(_int_context())

        assert client.calls == 2
        assert result.repair_method == "llm"
        assert repairer.stats.retries == 1
        assert repairer.stats.llm_calls == 1          # one logical request
        assert repairer.stats.transport_attempts == 2  # two physical attempts

    def test_retry_on_500_then_success(self):
        client = FakeClient([LLMHTTPError(503, "unavailable"), _resp((3, 1.0))])
        repairer = _repairer(client)

        result = repairer.repair(_int_context())

        assert client.calls == 2
        assert result.repair_method == "llm"

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_no_retry_on_permanent_statuses(self, status):
        client = FakeClient([LLMHTTPError(status, "nope")])
        repairer = _repairer(client)

        result = repairer.repair(_int_context())

        assert client.calls == 1
        assert repairer.stats.retries == 0
        assert result.repair_method == "llm->mock (request_failed)"

    def test_retries_are_bounded_by_max_retries(self):
        client = FakeClient([LLMHTTPError(429, "slow down")])  # always fails
        repairer = _repairer(client, config=_config(max_retries=2))

        result = repairer.repair(_int_context())

        assert client.calls == 3  # first attempt + 2 retries
        assert repairer.stats.retries == 2
        assert result.repair_method == "llm->mock (request_failed)"

    def test_max_retries_zero_means_one_attempt(self):
        client = FakeClient([LLMHTTPError(429, "slow down")])
        repairer = _repairer(client, config=_config(max_retries=0))

        repairer.repair(_int_context())

        assert client.calls == 1

    def test_backoff_is_exponential(self, network_guard):
        client = FakeClient([LLMHTTPError(429, "slow down")])
        repairer = _repairer(client, config=_config(max_retries=2, retry_backoff_s=0.5))

        repairer.repair(_int_context())

        assert network_guard.sleeps == [0.5, 1.0]

    def test_zero_backoff_does_not_sleep(self, network_guard):
        client = FakeClient([LLMHTTPError(429, "x"), _resp((3, 1.0))])
        repairer = _repairer(client, config=_config(retry_backoff_s=0.0))

        repairer.repair(_int_context())

        assert network_guard.sleeps == []

    @pytest.mark.parametrize(
        "status,expected",
        [(429, True), (500, True), (503, True), (599, True), (408, True),
         (400, False), (401, False), (403, False), (404, False), (None, False)],
    )
    def test_is_retryable_status(self, status, expected):
        assert is_retryable_status(status) is expected


# ═══════════════════════════════════════════════════════════════════════════════
# Stats
# ═══════════════════════════════════════════════════════════════════════════════


class TestStats:
    def test_counts_across_a_mixed_batch(self):
        client = FakeClient([
            _resp((3, 1.0)),                      # success
            "not json",                           # malformed
            LLMTimeoutError("too slow"),          # timeout
            _resp((5, 1.0)),                      # success
        ])
        repairer = _repairer(client)

        results = [repairer.repair(_int_context(row_id=i)) for i in range(4)]
        stats = repairer.stats

        assert [r.repair_method for r in results] == [
            "llm",
            "llm->mock (malformed_response)",
            "llm->mock (timeout)",
            "llm",
        ]
        assert stats.cells == 4
        assert stats.llm_calls == 4
        assert stats.successes == 2
        assert stats.fallbacks == 2
        assert stats.fallback_reasons == {"malformed_response": 1, "timeout": 1}
        assert stats.fallback_rate == pytest.approx(0.5)

    def test_to_dict_shape(self):
        stats = LLMStats()
        stats.record_fallback(FallbackReason.TIMEOUT)
        stats.record_fallback(FallbackReason.TIMEOUT)
        stats.cells = 2

        payload = stats.to_dict()

        assert payload["fallbacks"] == 2
        assert payload["fallback_reasons"] == {"timeout": 2}
        assert payload["fallback_rate"] == pytest.approx(1.0)
        assert set(payload) == {
            "cells", "llm_calls", "transport_attempts", "retries", "cache_hits",
            "successes", "fallbacks", "fallback_reasons", "fallback_rate",
        }
        assert json.dumps(payload)  # serialisable

    def test_reset_zeroes_everything(self):
        stats = LLMStats(cells=3, llm_calls=3, successes=1)
        stats.record_fallback(FallbackReason.REQUEST_FAILED)

        stats.reset()

        assert stats.to_dict()["cells"] == 0
        assert stats.fallback_reasons == {}
        assert stats.fallback_rate == 0.0

    def test_empty_stats_rate_is_zero(self):
        assert LLMStats().fallback_rate == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# repair_batch — drop-in compatibility with MockRepairer
# ═══════════════════════════════════════════════════════════════════════════════


def _batch_profile() -> DatasetProfile:
    columns = {
        "age": ColumnProfile(
            name="age", kind=ColumnKind.NUMERICAL, dtype_clean="int64",
            dtype_dirty="float64", missing_clean=0, missing_dirty=1,
            n_unique_clean=3, n_unique_dirty=3, numerical_stats=_num_stats(),
        ),
        "job": ColumnProfile(
            name="job", kind=ColumnKind.CATEGORICAL, dtype_clean="object",
            dtype_dirty="object", missing_clean=0, missing_dirty=0,
            n_unique_clean=2, n_unique_dirty=3,
            categorical_stats=CategoricalStats(
                value_counts={"Welder": 2, "Baker": 1},
                top_values=["Welder", "Baker"], n_rare=0,
            ),
        ),
    }
    return DatasetProfile(
        name="batch_ds",
        clean_path=Path("datasets/batch_ds/clean.csv"),
        dirty_path=Path("datasets/batch_ds/dirty.csv"),
        n_rows=3, n_cols=2, column_names=list(columns), columns=columns,
    )


def _batch_dirty_df() -> pd.DataFrame:
    return pd.DataFrame({"age": [25.0, 999.0, 31.0], "job": ["Weldrr", "Welder", "Baker"]})


def _batch_records() -> List[DetectionRecord]:
    return [
        DetectionRecord(
            dataset="batch_ds", row_id=0, column="job", dirty_value="Weldrr",
            error_type=ErrorType.NOVEL_CATEGORY.value, detection_score=0.9,
            detection_method="novel_category", reason="not in vocabulary",
        ),
        DetectionRecord(
            dataset="batch_ds", row_id=1, column="age", dirty_value=999.0,
            error_type=ErrorType.OUT_OF_RANGE.value, detection_score=0.95,
            detection_method="range_check", reason="above max",
        ),
    ]


class TestRepairBatch:
    def test_signatures_match_the_mock_repairer(self):
        for method in ("repair", "repair_batch"):
            llm_sig = inspect.signature(getattr(LLMRepairer, method))
            mock_sig = inspect.signature(getattr(MockRepairer, method))
            assert list(llm_sig.parameters) == list(mock_sig.parameters), method

    def test_no_clean_df_parameter_anywhere(self):
        # Leakage guard: there is no way to hand clean data to this repairer.
        for method in ("repair", "repair_batch", "build_payload"):
            params = inspect.signature(getattr(LLMRepairer, method)).parameters
            assert not any("clean" in p for p in params)

    def test_output_aligns_one_to_one_with_input(self):
        client = FakeClient([_resp(("Welder", 1.0)), _resp((5, 1.0))])
        repairer = _repairer(client)
        records = _batch_records()

        results = repairer.repair_batch(records, _batch_dirty_df(), _batch_profile())

        assert len(results) == len(records)
        assert [(r.row_id, r.column) for r in results] == [
            (rec.row_id, rec.column) for rec in records
        ]
        assert [r.repair_method for r in results] == ["llm", "llm"]
        assert repairer.stats.cells == 2

    def test_field_shape_matches_mock_repairer_output(self):
        client = FakeClient([_resp(("Welder", 1.0)), _resp((5, 1.0))])
        records, dirty_df, profile = _batch_records(), _batch_dirty_df(), _batch_profile()

        llm_results = _repairer(client).repair_batch(records, dirty_df, profile)
        mock_results = MockRepairer(seed=42, top_k=3).repair_batch(records, dirty_df, profile)

        assert len(llm_results) == len(mock_results)
        for llm_r, mock_r in zip(llm_results, mock_results):
            assert type(llm_r) is type(mock_r)
            assert set(llm_r.to_dict()) == set(mock_r.to_dict())
            assert (llm_r.dataset, llm_r.row_id, llm_r.column) == (
                mock_r.dataset, mock_r.row_id, mock_r.column
            )
            assert llm_r.dirty_value == mock_r.dirty_value or (
                pd.isna(llm_r.dirty_value) and pd.isna(mock_r.dirty_value)
            )

    def test_batch_ordering_survives_a_mid_batch_fallback(self):
        client = FakeClient(["garbage", _resp((5, 1.0))])
        repairer = _repairer(client)
        records = _batch_records()

        results = repairer.repair_batch(records, _batch_dirty_df(), _batch_profile())

        assert [r.repair_method for r in results] == [
            "llm->mock (malformed_response)", "llm"
        ]
        assert [r.column for r in results] == ["job", "age"]

    def test_empty_batch_returns_empty_list(self):
        repairer = _repairer(FakeClient())

        results = repairer.repair_batch([], _batch_dirty_df(), _batch_profile())

        assert results == []
        assert repairer.stats.cells == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Leakage — the captured request body must contain no ground truth
# ═══════════════════════════════════════════════════════════════════════════════


def _leak_dirty_df() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [25.0, 999.0, 31.0],
        "job": ["Weldrr", "Welder", "Baker"],
        "note": ["note_row0_ok", "note_row1_ok", "ROW2_NOTE_SENTINEL"],
    })


def _leak_clean_df() -> pd.DataFrame:
    """
    Ground truth whose every cell is deliberately disjoint from the dirty cell.

    Real clean/dirty pairs agree on most cells, which would make a
    "no clean value appears" assertion vacuously false. Making every clean
    value a distinct sentinel lets the assertion be exact: if any ground-truth
    cell reached the payload, the test fails.
    """
    return pd.DataFrame({
        "age": [404401, 404402, 404403],
        "job": ["CLEANJOB_ALPHA", "CLEANJOB_BETA", "CLEANJOB_GAMMA"],
        "note": ["CLEANNOTE_ALPHA", "CLEANNOTE_BETA", "CLEANNOTE_GAMMA"],
    })


def _leak_profile() -> DatasetProfile:
    columns = {
        "age": ColumnProfile(
            name="age", kind=ColumnKind.NUMERICAL, dtype_clean="int64",
            dtype_dirty="float64", missing_clean=0, missing_dirty=0,
            n_unique_clean=3, n_unique_dirty=3, numerical_stats=_num_stats(),
        ),
        "job": ColumnProfile(
            name="job", kind=ColumnKind.CATEGORICAL, dtype_clean="object",
            dtype_dirty="object", missing_clean=0, missing_dirty=0,
            n_unique_clean=3, n_unique_dirty=3,
            categorical_stats=CategoricalStats(
                value_counts={"Welder": 1, "Baker": 1}, top_values=["Welder"], n_rare=0,
            ),
        ),
        "note": ColumnProfile(
            name="note", kind=ColumnKind.CATEGORICAL, dtype_clean="object",
            dtype_dirty="object", missing_clean=0, missing_dirty=0,
            n_unique_clean=3, n_unique_dirty=3,
            categorical_stats=CategoricalStats(
                value_counts={"note_row0_ok": 1}, top_values=["note_row0_ok"], n_rare=0,
            ),
        ),
    }
    return DatasetProfile(
        name="leak_ds",
        clean_path=Path("datasets/leak_ds/clean.csv"),
        dirty_path=Path("datasets/leak_ds/dirty.csv"),
        n_rows=3, n_cols=3, column_names=list(columns), columns=columns,
    )


class TestLeakage:
    def _capture(self) -> FakeClient:
        """Repair row 0 of the leak dataset, passing only dirty data."""
        client = FakeClient([_resp(("Welder", 1.0))])
        repairer = _repairer(client)
        record = DetectionRecord(
            dataset="leak_ds", row_id=0, column="job", dirty_value="Weldrr",
            error_type=ErrorType.NOVEL_CATEGORY.value, detection_score=0.9,
            detection_method="novel_category", reason="not in vocabulary",
        )
        repairer.repair_batch([record], _leak_dirty_df(), _leak_profile())
        assert client.calls == 1
        return client

    def test_no_clean_value_reaches_the_request_body(self):
        client = self._capture()
        body = client.payload_text

        for column in _leak_clean_df().columns:
            for value in _leak_clean_df()[column]:
                assert str(value) not in body, f"clean value {value!r} leaked"

    def test_no_other_rows_data_reaches_the_request_body(self):
        client = self._capture()
        body = client.payload_text

        # Row 2's `note` value can only appear if some other row's record was
        # sent; row 0's own note context is expected and asserted below.
        assert "ROW2_NOTE_SENTINEL" not in body
        assert "note_row1_ok" not in body

    def test_the_payload_does_contain_the_permitted_context(self):
        # Positive control: without this the leak assertions could pass on an
        # empty payload.
        client = self._capture()
        body = client.payload_text

        assert "Weldrr" in body          # the dirty value
        assert "job" in body             # the column name
        assert "leak_ds" in body         # the dataset name
        assert "note_row0_ok" in body    # the same row's dirty context
        assert "Baker" in body           # vocabulary mined from dirty data

    def test_payload_is_built_from_the_context_only(self):
        # build_payload takes a RepairContext and nothing else.
        client = FakeClient([_resp((3, 1.0))])
        repairer = _repairer(client)

        payload = repairer.build_payload(_int_context())

        assert set(payload) == {"model", "messages", "temperature", "max_tokens"}
        assert len(payload["messages"]) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# prompts.parse_llm_response_detailed — the strict parser in isolation
# ═══════════════════════════════════════════════════════════════════════════════


class TestStrictParser:
    def test_success_returns_no_reason(self):
        candidates, reason = parse_llm_response_detailed(
            _resp((3, 0.7), (5, 0.3)), _int_context()
        )

        assert reason is None
        assert [c.value for c in candidates] == [3, 5]

    @pytest.mark.parametrize(
        "body,expected",
        [
            ("", PARSE_MALFORMED_RESPONSE),
            ("   ", PARSE_MALFORMED_RESPONSE),
            ("not json", PARSE_MALFORMED_RESPONSE),
            ('{"candidates": "three"}', PARSE_MALFORMED_RESPONSE),
            ('{"candidates": [3, 5]}', PARSE_MALFORMED_RESPONSE),
            ('{"candidates": [{"value": 3}]}', PARSE_MALFORMED_RESPONSE),
            ('{"nope": 1}', PARSE_MALFORMED_RESPONSE),
            ('{"candidates": []}', PARSE_EMPTY_CANDIDATES),
            ('{"candidates": [{"value": 3, "probability": -1}]}',
             PARSE_INVALID_PROBABILITIES),
        ],
    )
    def test_reason_codes(self, body, expected):
        candidates, reason = parse_llm_response_detailed(body, _int_context())

        assert candidates == []
        assert reason == expected

    def test_non_string_input_is_malformed(self):
        candidates, reason = parse_llm_response_detailed(None, _int_context())  # type: ignore[arg-type]
        assert (candidates, reason) == ([], PARSE_MALFORMED_RESPONSE)

    def test_boolean_probability_is_invalid(self):
        _, reason = parse_llm_response_detailed(
            '{"candidates": [{"value": 3, "probability": true}]}', _int_context()
        )
        assert reason == PARSE_INVALID_PROBABILITIES

    def test_container_value_is_malformed(self):
        _, reason = parse_llm_response_detailed(
            '{"candidates": [{"value": [3], "probability": 1.0}]}', _cat_context()
        )
        assert reason == PARSE_MALFORMED_RESPONSE

    def test_top_k_of_one_keeps_the_best(self):
        candidates, reason = parse_llm_response_detailed(
            _resp((3, 0.3), (5, 0.7)), _int_context(), top_k=1
        )

        assert reason is None
        assert len(candidates) == 1
        assert candidates[0].value == 5
        assert candidates[0].probability == pytest.approx(1.0)

    def test_sum_tolerance_is_configurable(self):
        body = _resp((3, 0.4), (5, 0.3))  # sums to 0.7

        strict = parse_llm_response_detailed(body, _int_context())
        lenient = parse_llm_response_detailed(body, _int_context(), sum_tolerance=0.5)

        assert strict[1] == PARSE_INVALID_PROBABILITIES
        assert lenient[1] is None

    def test_bool_dtype_casting(self):
        ctx = _cat_context(dtype="bool")

        candidates, reason = parse_llm_response_detailed(
            _resp(("true", 0.6), ("false", 0.4)), ctx
        )

        assert reason is None
        assert candidates[0].value is True

        _, bad_reason = parse_llm_response_detailed(_resp(("maybe", 1.0)), ctx)
        assert bad_reason == PARSE_MALFORMED_RESPONSE


class TestLenientParserUnchanged:
    """The original parse_llm_response must keep its old signature/behaviour."""

    def test_signature_is_unchanged(self):
        params = inspect.signature(parse_llm_response).parameters
        assert list(params) == ["response_text", "context", "top_k"]

    def test_still_returns_a_bare_list(self):
        out = parse_llm_response(_resp((3, 0.8), (5, 0.2)), _int_context())
        assert isinstance(out, list)
        assert len(out) == 2

    def test_still_drops_bad_candidates_instead_of_rejecting(self):
        body = '{"candidates": [{"value": 3, "probability": -0.1}, ' \
               '{"value": 5, "probability": 0.9}]}'

        lenient = parse_llm_response(body, _int_context())
        _, strict_reason = parse_llm_response_detailed(body, _int_context())

        assert len(lenient) == 1                                  # old behaviour
        assert strict_reason == PARSE_INVALID_PROBABILITIES       # new behaviour

    def test_invalid_json_still_returns_empty_list(self):
        assert parse_llm_response("not json at all", _int_context()) == []
