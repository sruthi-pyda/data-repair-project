"""
src/repair/llm_repair.py
-------------------------
Real-LLM repair-candidate generator (Phase 12, Step 1).

PURPOSE
=======
`LLMRepairer` is a **drop-in replacement for `MockRepairer`**: identical
`repair(context)` / `repair_batch(records, dirty_df, profile)` signatures and
identical `RepairResult` output type. Everything downstream of Phase 4
(uncertainty → calibration → risk → decision → evaluation) is repairer-agnostic
by design, so swapping the repairer changes *what `p` is calibrated from* and
nothing else.

The project is called "Risk-Aware **LLM-Assisted** Data Repair"; until this
module exists there is not one real model call in it. The research claim does
not change — it is still about calibration and cost-sensitive decisions.

PROVIDER
========
The client is **OpenAI-compatible and configured by environment variable**, not
hard-wired to a vendor:

    LLM_API_KEY  = ...
    LLM_BASE_URL = https://api.x.ai/v1     # xAI/Grok; omit for OpenAI
    LLM_MODEL    = grok-...                # exact id from the provider console

Transport is chosen at run time: the `openai` SDK if it is importable, otherwise
a dependency-free `urllib` POST to `{base_url}/chat/completions`. Both sit
behind the single `LLMTransport` interface, so tests inject a fake and **never
touch the network**.

API-KEY HANDLING
================
The key is read from the environment only. It is never read into a log line, a
`repr`, a cache file, a `to_dict()` or an exception message: `LLMConfig` has a
hand-written `__repr__` that prints `api_key='<set>'` / `'<unset>'`, and every
message that could conceivably echo the key is passed through `_redact()`.

KEY ROTATION (a throughput problem, not a feature)
==================================================
A full experiment is 392 (breast_cancer) + 7,093 (smartfactory) + 5,829
(adult) = 13,314 calls, and Groq rate-limits **per key**. Backing off on a 429
while a spare key sits idle wastes wall-clock time for nothing, so:

    LLM_API_KEYS = key1,key2,key3,...        (comma-separated pool)
    LLM_API_KEY  = key1                      (single key; the pool falls back
                                              to [api_key])

`LLMConfig.key_pool` is the de-duplicated list of *usable* keys (placeholders
dropped). On HTTP 429 for the active key the repairer advances to the next key
in the pool and retries **immediately** — no sleep, and the retry budget is not
consumed, because a fresh key is not the same request being re-sent. Only when
every key has been rate-limited **within this one request** does it fall back
to the existing exponential backoff, and then, if that also fails, to
`MockRepairer` for that cell. The rotated-to index persists across cells, so a
key that is already throttled is not tried again for the next cell.

`LLMStats.rate_limited` counts 429 responses and `LLMStats.key_rotations`
counts the advances, so a run can report "we were throttled 240 times and
rotated 238 times" instead of silently taking four times as long. Keys
themselves never appear in the stats, the logs or any artifact.

DATA LEAKAGE PREVENTION (CLAUDE.md §3 — non-negotiable)
=======================================================
The outgoing request body is built **only** from `RepairContext`, which is
constructed from dirty data plus aggregate profile statistics. The payload
therefore carries: dataset name, column name, dtype/kind, the dirty value, the
same row's other *dirty* values, a vocabulary mined from the *dirty* column, and
aggregate column statistics.

Never sent: `clean.csv`, any per-cell clean value, any ground-truth label, any
other row's clean data, or the full dataset. `tests/test_llm_repair.py` asserts
this against the *captured request body*, not against the prompt builder's
intent.

FAILURE HANDLING
================
Fallback is **per cell and never fatal**. Any failure — unconfigured key,
request failure, timeout, malformed or invalid response — increments a counter
and falls back to `MockRepairer` for that one cell. The fallback is visible in
the predictions CSV through `RepairResult.repair_method`:

    "llm"                            → the model answered and validated
    "llm->mock (request_failed)"     → this cell fell back, and why

An ASCII arrow is used deliberately so the value survives any CSV encoding.

DETERMINISM AND COST
====================
`temperature=0.0` by default, and an on-disk JSON cache keyed by
`sha256(model + system prompt + user prompt)` means a re-run of the same cells
costs nothing and reproduces exactly. `use_cache=False` forces fresh calls.
Retries happen only on HTTP 429 and 5xx, with exponential backoff; 400/401/403
are never retried (they will never succeed).

WORKED EXAMPLE
==============
    from src.repair.llm_repair import LLMConfig, LLMRepairer

    config = LLMConfig.from_env()          # reads .env / os.environ
    repairer = LLMRepairer(config=config, top_k=3, seed=42)

    results = repairer.repair_batch(records, dirty_df, profile)   # no clean_df

    print(results[0].repair_method)        # 'llm'  or  'llm->mock (timeout)'
    print(results[0].top_candidate)        # RepairCandidate(value=3, probability=0.71)
    print(repairer.stats.to_dict())
    # {'cells': 54, 'llm_calls': 54, 'cache_hits': 0, 'successes': 51,
    #  'fallbacks': 3, 'fallback_reasons': {'malformed_response': 3}, ...}

If no key is configured every cell falls back and the run still completes, with
`repair_method == 'llm->mock (not_configured)'` on every record — the pipeline
degrades to mock mode loudly rather than crashing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

import pandas as pd

from src.data.schema import DatasetProfile
from src.detection.detector import DetectionRecord
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
    SYSTEM_PROMPT,
    build_repair_prompt,
    parse_llm_response_detailed,
)

logger = logging.getLogger(__name__)


# Default model id. Deliberately a cheap, widely available OpenAI-compatible
# model; the real id comes from LLM_MODEL in the environment.
DEFAULT_MODEL: str = "gpt-4o-mini"

# Used only by the urllib transport when LLM_BASE_URL is not set.
DEFAULT_BASE_URL: str = "https://api.openai.com/v1"

# Values that look like a key but are not one. `.env.example` ships
# "your-key-here", and a half-filled .env must count as *unconfigured* rather
# than produce an authentication failure per cell.
PLACEHOLDER_KEYS: Tuple[str, ...] = (
    "your-key-here",
    "your_key_here",
    "your-api-key",
    "your_api_key",
    "changeme",
    "change-me",
    "none",
    "todo",
    "xxx",
    "sk-...",
)

# HTTP statuses worth retrying: rate limiting and server-side faults.
RETRYABLE_STATUSES: Tuple[int, ...] = (408, 409, 425, 429)

# The one status that rotation reacts to: "this key is throttled right now".
# 5xx is a provider-side fault and affects every key, so rotating on it would
# burn the whole pool on one outage.
RATE_LIMIT_STATUS: int = 429


def is_usable_key(key: Optional[str]) -> bool:
    """
    True when `key` looks like a real credential rather than a placeholder.

    A blank value, or one of `PLACEHOLDER_KEYS` / anything starting ``your-``
    or ``your_`` (as shipped in `.env.example`), counts as *absent*, so a
    half-filled `.env` falls back cleanly instead of burning one
    authentication failure per cell.
    """
    text = (key or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if lowered in PLACEHOLDER_KEYS:
        return False
    return not (lowered.startswith("your-") or lowered.startswith("your_"))


def _split_keys(raw: Optional[str]) -> Tuple[str, ...]:
    """Parse ``LLM_API_KEYS`` — a comma-separated pool — order preserved."""
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


# ═══════════════════════════════════════════════════════════════════════════════
# Errors
# ═══════════════════════════════════════════════════════════════════════════════


class LLMError(Exception):
    """Base class for every transport-level failure raised by this module."""


class LLMHTTPError(LLMError):
    """
    A non-2xx HTTP response from the provider.

    `status_code` drives the retry decision: 429 and 5xx are retried, every
    other status (notably 400/401/403) is permanent and falls back immediately.
    """

    def __init__(self, status_code: int, message: str = "") -> None:
        self.status_code = int(status_code)
        self.message = message or ""
        super().__init__(f"HTTP {self.status_code}: {self.message}")


class LLMTimeoutError(LLMError):
    """The request exceeded `LLMConfig.timeout_s`."""


class LLMTransportError(LLMError):
    """A connection-level failure (DNS, TLS, refused socket, bad payload)."""


def is_retryable_status(status_code: Optional[int]) -> bool:
    """True for 429-style throttling and 5xx server faults; False otherwise."""
    if status_code is None:
        return False
    return status_code in RETRYABLE_STATUSES or 500 <= int(status_code) <= 599


# ═══════════════════════════════════════════════════════════════════════════════
# Redaction
# ═══════════════════════════════════════════════════════════════════════════════


def _redact(text: Any, secret: Optional[str]) -> str:
    """
    Return `text` as a string with `secret` replaced by '<redacted>'.

    Belt-and-braces: the key is never deliberately placed in a message, but a
    provider could echo an Authorization header back in an error body, so every
    message that reaches a log or an exception goes through here first.
    """
    out = "" if text is None else str(text)
    if secret and len(secret) >= 4:
        out = out.replace(secret, "<redacted>")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(repr=False)
class LLMConfig:
    """
    Connection and sampling configuration for an OpenAI-compatible endpoint.

    The API key lives here but is never printed: `__repr__` (and therefore
    `str`) shows `api_key='<set>'` or `'<unset>'`, and `to_dict()` exposes
    `api_key_status` instead of the value, so the config is safe to log, to dump
    into a results bundle, and to include in an exception message.

    Fields
    ------
    api_key : str | None
        From LLM_API_KEY. Never logged.
    api_keys : tuple[str, ...]
        From LLM_API_KEYS (comma-separated), for per-key rate-limit rotation.
        Empty means "the pool is just `api_key`". Never logged.
    base_url : str | None
        From LLM_BASE_URL. `None` means the provider default (OpenAI).
        xAI/Grok example: https://api.x.ai/v1
    model : str
        From LLM_MODEL.
    temperature : float
        0.0 by default, for determinism and reproducibility.
    max_tokens, timeout_s : int, float
        Per-request limits.
    max_retries : int
        Extra attempts after the first, for 429/5xx only (2 → 3 attempts max).
    retry_backoff_s : float
        Base delay; attempt *n* waits `retry_backoff_s * 2**(n-1)` seconds.
        0.0 disables sleeping (used by tests).
    cache_dir : str
        Directory for the on-disk response cache.
    use_cache : bool
        Default for `LLMRepairer(use_cache=...)`. False forces fresh calls
        (``--no-llm-cache`` on the experiment runner).
    """

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    # 1536, not 512: reasoning models (e.g. Groq's openai/gpt-oss-120b) spend
    # most of their completion budget on hidden reasoning tokens and emit the
    # JSON last, so a tight cap truncates the structure mid-object and the
    # response is rejected as malformed. Measured on 12 breast_cancer cells:
    # 512 -> 17% fallback, 1536 -> 8%.
    max_tokens: int = 1536
    timeout_s: float = 30.0
    max_retries: int = 2
    retry_backoff_s: float = 0.5
    cache_dir: str = ".cache/llm"
    # Declared last so every existing positional construction keeps working.
    api_keys: Tuple[str, ...] = ()
    use_cache: bool = True

    # ── Construction ────────────────────────────────────────────────────────

    @classmethod
    def from_env(cls, dotenv_path: Union[str, Path, None] = ".env") -> "LLMConfig":
        """
        Build a config from `.env` (if present) plus the process environment.

        `python-dotenv` is used when importable; otherwise a tiny built-in
        parser reads `KEY=VALUE` lines. Real environment variables always win
        over `.env`, matching dotenv's `override=False` default.

        The key is read straight into the dataclass field. It is not logged,
        echoed, or written anywhere by this method.

        ``LLM_API_KEYS`` (comma-separated) populates the rotation pool. When it
        is absent the pool falls back to ``[LLM_API_KEY]``; when only
        ``LLM_API_KEYS`` is set, its first entry also becomes ``api_key`` so a
        single-key caller (and the default transport) still works.
        """
        env = _load_dotenv_values(dotenv_path)

        def _get(name: str) -> Optional[str]:
            val = os.environ.get(name)
            if val is None:
                val = env.get(name)
            if val is None:
                return None
            val = val.strip()
            return val or None

        pool = _split_keys(_get("LLM_API_KEYS"))
        single = _get("LLM_API_KEY") or (pool[0] if pool else None)

        return cls(
            api_key=single,
            api_keys=pool or ((single,) if single else ()),
            use_cache=_as_bool(_get("LLM_USE_CACHE"), True),
            base_url=_get("LLM_BASE_URL"),
            model=_get("LLM_MODEL") or DEFAULT_MODEL,
            temperature=_as_float(_get("LLM_TEMPERATURE"), 0.0),
            max_tokens=int(_as_float(_get("LLM_MAX_TOKENS"), 1536)),
            timeout_s=_as_float(_get("LLM_TIMEOUT_S"), 30.0),
            max_retries=int(_as_float(_get("LLM_MAX_RETRIES"), 2)),
            retry_backoff_s=_as_float(_get("LLM_RETRY_BACKOFF_S"), 0.5),
            cache_dir=_get("LLM_CACHE_DIR") or ".cache/llm",
        )

    # ── Introspection ───────────────────────────────────────────────────────

    @property
    def key_pool(self) -> Tuple[str, ...]:
        """
        The usable keys, de-duplicated, in order — the rotation pool.

        ``api_keys`` wins when it is set; otherwise the pool is ``[api_key]``.
        Placeholders and blanks are dropped, so a half-filled `.env` yields an
        empty pool (i.e. *unconfigured*) rather than a key that cannot work.
        """
        raw = self.api_keys if self.api_keys else ((self.api_key,) if self.api_key else ())
        seen: Dict[str, None] = {}
        for key in raw:
            text = (key or "").strip()
            if is_usable_key(text) and text not in seen:
                seen[text] = None
        return tuple(seen)

    @property
    def n_api_keys(self) -> int:
        """Size of the rotation pool (never the keys themselves)."""
        return len(self.key_pool)

    @property
    def primary_api_key(self) -> Optional[str]:
        """First usable key, or ``api_key`` when nothing in the pool is usable."""
        pool = self.key_pool
        return pool[0] if pool else self.api_key

    @property
    def is_configured(self) -> bool:
        """
        True when at least one usable key is present.

        A missing, blank or placeholder key (e.g. the shipped
        "your-key-here") counts as *unconfigured*, so the repairer falls back
        with `FallbackReason.NOT_CONFIGURED` instead of burning one
        authentication failure per cell. With a pool, one usable key is enough.
        """
        return bool(self.key_pool)

    @property
    def api_key_status(self) -> str:
        """'<set>' or '<unset>' — the only key information this module reveals."""
        present = bool((self.api_key or "").strip()) or bool(self.key_pool)
        return "<set>" if present else "<unset>"

    @property
    def effective_base_url(self) -> str:
        """Base URL actually used by the urllib transport (no trailing slash)."""
        return (self.base_url or DEFAULT_BASE_URL).rstrip("/")

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable view with the key redacted."""
        return {
            "api_key_status": self.api_key_status,
            "is_configured": self.is_configured,
            "n_api_keys": self.n_api_keys,
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_s": self.timeout_s,
            "max_retries": self.max_retries,
            "retry_backoff_s": self.retry_backoff_s,
            "cache_dir": self.cache_dir,
            "use_cache": self.use_cache,
        }

    def __repr__(self) -> str:  # str() falls back to this too
        return (
            "LLMConfig("
            f"api_key={self.api_key_status!r}, "
            f"n_api_keys={self.n_api_keys!r}, "
            f"base_url={self.base_url!r}, "
            f"model={self.model!r}, "
            f"temperature={self.temperature!r}, "
            f"max_tokens={self.max_tokens!r}, "
            f"timeout_s={self.timeout_s!r}, "
            f"max_retries={self.max_retries!r}, "
            f"cache_dir={self.cache_dir!r})"
        )


def _as_float(raw: Optional[str], default: float) -> float:
    """Parse a numeric environment value, falling back to `default`."""
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning("Ignoring non-numeric environment value; using %s.", default)
        return default


def _as_bool(raw: Optional[str], default: bool) -> bool:
    """Parse a boolean environment value ('0', 'false', 'no', 'off' → False)."""
    if raw is None:
        return default
    text = raw.strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    logger.warning("Ignoring non-boolean environment value; using %s.", default)
    return default


def _load_dotenv_values(dotenv_path: Union[str, Path, None]) -> Dict[str, str]:
    """
    Read `KEY=VALUE` pairs from a .env file without mutating `os.environ`.

    Uses `dotenv.dotenv_values` when python-dotenv is installed, else a minimal
    parser (comments, blank lines and surrounding quotes handled). Returns an
    empty dict when the file is missing or unreadable. Values are never logged.
    """
    if dotenv_path is None:
        return {}
    path = Path(dotenv_path)
    if not path.is_file():
        return {}

    try:
        from dotenv import dotenv_values  # type: ignore

        return {k: v for k, v in dotenv_values(str(path)).items() if v is not None}
    except ImportError:
        pass
    except Exception:  # pragma: no cover - defensive
        logger.debug("dotenv_values failed for %s; using the built-in parser.", path)

    values: Dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            val = val.strip().strip('"').strip("'")
            values[key.strip()] = val
    except OSError:  # pragma: no cover - defensive
        logger.debug("Could not read %s; continuing with os.environ only.", path)
    return values


# ═══════════════════════════════════════════════════════════════════════════════
# Fallback reasons and statistics
# ═══════════════════════════════════════════════════════════════════════════════


class FallbackReason(str, Enum):
    """
    Why a single cell fell back from the LLM to `MockRepairer`.

    The value appears verbatim in `RepairResult.repair_method`, e.g.
    `"llm->mock (invalid_probabilities)"`, so a fallback is visible in the
    predictions CSV rather than silently indistinguishable from a real answer.
    """

    NOT_CONFIGURED = "not_configured"
    REQUEST_FAILED = "request_failed"
    MALFORMED_RESPONSE = "malformed_response"
    INVALID_PROBABILITIES = "invalid_probabilities"
    EMPTY_CANDIDATES = "empty_candidates"
    TIMEOUT = "timeout"


# Parser reason codes (plain strings, so prompts.py need not import this
# module) map 1:1 onto the enum.
_PARSE_REASON_MAP: Dict[str, FallbackReason] = {
    PARSE_MALFORMED_RESPONSE: FallbackReason.MALFORMED_RESPONSE,
    PARSE_INVALID_PROBABILITIES: FallbackReason.INVALID_PROBABILITIES,
    PARSE_EMPTY_CANDIDATES: FallbackReason.EMPTY_CANDIDATES,
}


@dataclass
class LLMStats:
    """
    Counters for one `LLMRepairer` instance.

    Fields
    ------
    cells : int
        Cells passed to `repair()`.
    llm_calls : int
        Logical requests sent to the transport (cache misses only).
    transport_attempts : int
        Physical attempts, including retries. `transport_attempts - llm_calls`
        is the retry volume.
    retries : int
        Retried attempts after a backoff (429/5xx only). A rotation retry is
        NOT counted here: it consumes no backoff and no retry budget.
    rate_limited : int
        HTTP 429 responses received, across every key.
    key_rotations : int
        Times the active key advanced to a spare because of a 429.
        ``rate_limited - key_rotations`` is roughly "429s we had to wait out
        because the whole pool was throttled".
    cache_hits : int
        Cells answered from the on-disk cache without a transport call.
    successes : int
        Cells where the model answered and the response validated.
    fallbacks : int
        Cells that fell back to `MockRepairer`.
    fallback_reasons : dict[str, int]
        Fallback counts keyed by `FallbackReason.value`.
    """

    cells: int = 0
    llm_calls: int = 0
    transport_attempts: int = 0
    retries: int = 0
    cache_hits: int = 0
    successes: int = 0
    fallbacks: int = 0
    fallback_reasons: Dict[str, int] = field(default_factory=dict)
    # Rotation counters, declared last so positional construction is unchanged.
    rate_limited: int = 0
    key_rotations: int = 0

    def record_fallback(self, reason: "FallbackReason") -> None:
        """Increment the total and the per-reason counter."""
        self.fallbacks += 1
        key = reason.value
        self.fallback_reasons[key] = self.fallback_reasons.get(key, 0) + 1

    @property
    def fallback_rate(self) -> float:
        """Fraction of cells that fell back; 0.0 when no cells were seen."""
        return (self.fallbacks / self.cells) if self.cells else 0.0

    def reset(self) -> None:
        """Zero every counter (used between datasets in one process)."""
        self.cells = 0
        self.llm_calls = 0
        self.transport_attempts = 0
        self.retries = 0
        self.cache_hits = 0
        self.successes = 0
        self.fallbacks = 0
        self.fallback_reasons = {}
        self.rate_limited = 0
        self.key_rotations = 0

    def to_dict(self, include_rotation: bool = False) -> Dict[str, Any]:
        """
        Serialisable counters.

        ``include_rotation`` adds ``rate_limited`` and ``key_rotations``. It
        defaults to False because the nine-key shape below is a published
        schema that `tests/test_llm_repair.py` pins exactly; every *new*
        consumer (the pipeline bundle, `run_metadata.json`, the dashboard)
        asks for the full view.
        """
        payload = {
            "cells": self.cells,
            "llm_calls": self.llm_calls,
            "transport_attempts": self.transport_attempts,
            "retries": self.retries,
            "cache_hits": self.cache_hits,
            "successes": self.successes,
            "fallbacks": self.fallbacks,
            "fallback_reasons": dict(self.fallback_reasons),
            "fallback_rate": round(self.fallback_rate, 4),
        }
        if include_rotation:
            payload["rate_limited"] = self.rate_limited
            payload["key_rotations"] = self.key_rotations
        return payload


# ═══════════════════════════════════════════════════════════════════════════════
# Transport interface and implementations
# ═══════════════════════════════════════════════════════════════════════════════


@runtime_checkable
class LLMTransport(Protocol):
    """
    The one seam between this module and the network.

    `payload` is an OpenAI chat-completions request body — exactly what would go
    over the wire — so a fake transport in a test can assert on the real thing.
    Implementations return the assistant message **text** and raise
    `LLMHTTPError` / `LLMTimeoutError` / `LLMTransportError` on failure.

    `set_api_key` is **optional**: key rotation calls it before every attempt
    when the transport defines it (`LLMRepairer._apply_active_key` duck-types
    it), so a transport that ignores rotation — or a fake from an older test —
    needs no change. The key stays out of `chat_completion`'s arguments on
    purpose: the captured payload is asserted on in leakage tests and must
    never contain a credential.
    """

    name: str

    def chat_completion(self, payload: Dict[str, Any], timeout_s: float) -> str:
        ...  # pragma: no cover - protocol


class OpenAISDKTransport:
    """
    Transport backed by the `openai` SDK (works for any compatible endpoint).

    The client is constructed lazily on the first call, so merely building an
    `LLMRepairer` never instantiates an HTTP client.
    """

    name = "openai-sdk"

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._api_key = config.primary_api_key
        self._client: Any = None

    def set_api_key(self, api_key: Optional[str]) -> None:
        """
        Switch the credential (key rotation).

        The SDK client binds its key at construction, so a change discards the
        cached client and the next call builds a new one. A no-op when the key
        is unchanged, so the common path still reuses one client.
        """
        if api_key != self._api_key:
            self._api_key = api_key
            self._client = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI  # imported lazily: optional dependency

            kwargs: Dict[str, Any] = {"api_key": self._api_key}
            if self._config.base_url:
                kwargs["base_url"] = self._config.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def chat_completion(self, payload: Dict[str, Any], timeout_s: float) -> str:
        key = self._api_key
        try:
            client = self._ensure_client()
            response = client.chat.completions.create(timeout=timeout_s, **payload)
        except Exception as exc:  # noqa: BLE001 - normalised below
            raise _normalise_sdk_exception(exc, key) from None
        try:
            text = response.choices[0].message.content
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise LLMTransportError(
                f"Unexpected SDK response shape: {_redact(exc, key)}"
            ) from None
        return text or ""


def _normalise_sdk_exception(exc: Exception, api_key: Optional[str]) -> LLMError:
    """Map an `openai` SDK exception onto this module's error hierarchy."""
    message = _redact(exc, api_key)
    name = type(exc).__name__
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)

    if "Timeout" in name:
        return LLMTimeoutError(message)
    if isinstance(status, int):
        return LLMHTTPError(status, message)
    return LLMTransportError(f"{name}: {message}")


class UrllibTransport:
    """
    Dependency-free transport: a plain POST to `{base_url}/chat/completions`.

    Used when the `openai` SDK is not installed. `urllib` is imported inside
    the call so importing this module opens no sockets.
    """

    name = "urllib"

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._api_key = config.primary_api_key

    def set_api_key(self, api_key: Optional[str]) -> None:
        """Switch the credential (key rotation); the next request's
        ``Authorization`` header carries this key."""
        self._api_key = api_key

    def chat_completion(self, payload: Dict[str, Any], timeout_s: float) -> str:
        import socket
        import urllib.error
        import urllib.request

        key = self._api_key
        url = f"{self._config.effective_base_url}/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key or ''}",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:  # pragma: no cover - defensive
                detail = ""
            raise LLMHTTPError(int(exc.code), _redact(detail, key)) from None
        except socket.timeout:
            raise LLMTimeoutError(f"Request to {url} timed out after {timeout_s}s") from None
        except urllib.error.URLError as exc:
            if isinstance(getattr(exc, "reason", None), socket.timeout):
                raise LLMTimeoutError(
                    f"Request to {url} timed out after {timeout_s}s"
                ) from None
            raise LLMTransportError(_redact(exc, key)) from None
        except OSError as exc:
            raise LLMTransportError(_redact(exc, key)) from None

        try:
            data = json.loads(raw)
            return data["choices"][0]["message"]["content"] or ""
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise LLMTransportError(
                f"Unexpected response shape: {_redact(exc, key)}"
            ) from None


def _build_default_transport(config: LLMConfig) -> LLMTransport:
    """
    Pick a transport: the `openai` SDK when importable, else urllib.

    Tests patch this function to prove no real transport is ever constructed.
    """
    try:
        import openai  # noqa: F401 - probe only

        logger.debug("Using the openai SDK transport.")
        return OpenAISDKTransport(config)
    except ImportError:
        logger.debug("openai SDK not installed; using the urllib transport.")
        return UrllibTransport(config)


# ═══════════════════════════════════════════════════════════════════════════════
# On-disk response cache
# ═══════════════════════════════════════════════════════════════════════════════


class DiskCache:
    """
    JSON-file cache of raw model responses, keyed by `sha256(model + prompt)`.

    One file per key under `directory`. A hit short-circuits the transport
    entirely, which makes re-runs free and bit-for-bit reproducible. Only the
    model id, the prompt hash and the response text are stored — never the API
    key, and never any clean value (the prompt itself contains none).

    Every filesystem error degrades to "miss" / "not written": a broken cache
    must never break a run.
    """

    def __init__(self, directory: Union[str, Path] = ".cache/llm") -> None:
        self.directory = Path(directory)

    @staticmethod
    def make_key(model: str, prompt: str) -> str:
        """sha256 over the model id and the full prompt text."""
        digest = hashlib.sha256()
        digest.update((model or "").encode("utf-8"))
        digest.update(b"\n")
        digest.update((prompt or "").encode("utf-8"))
        return digest.hexdigest()

    def path_for(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> Optional[str]:
        """Return the cached response text, or None on miss/unreadable file."""
        path = self.path_for(key)
        try:
            if not path.is_file():
                return None
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            logger.debug("Ignoring unreadable cache entry %s.", path.name)
            return None
        response = payload.get("response")
        return response if isinstance(response, str) else None

    def set(self, key: str, response: str, model: str = "") -> None:
        """Write a cache entry; silently skip on any filesystem error."""
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.path_for(key).write_text(
                json.dumps({"key": key, "model": model, "response": response}, indent=2),
                encoding="utf-8",
            )
        except OSError:
            logger.debug("Could not write cache entry %s.", key[:12])


# ═══════════════════════════════════════════════════════════════════════════════
# LLMRepairer
# ═══════════════════════════════════════════════════════════════════════════════


class LLMRepairer:
    """
    Real-LLM repair-candidate generator — a drop-in for `MockRepairer`.

    Parameters
    ----------
    config : LLMConfig | None
        Connection settings. `None` reads them from `.env` / `os.environ`.
    fallback : MockRepairer | None
        Per-cell fallback repairer. `None` builds `MockRepairer(seed, top_k)`.
    top_k : int
        Maximum candidates per cell (also sent to the model in the prompt).
    seed : int
        Passed to the fallback repairer, keeping fallbacks deterministic.
    cache : DiskCache | str | Path | None
        `None` → `DiskCache(config.cache_dir)`. A path builds a cache there.
        Any object with `make_key`/`get`/`set` is accepted (tests use one).
    client : LLMTransport | None
        Injected transport. **Tests always pass one, so no test touches the
        network.** `None` builds the default transport lazily, on first use.
    use_cache : bool | None
        False disables cache reads and writes for this repairer. ``None``
        (the default) takes `LLMConfig.use_cache`, which is itself True
        unless ``LLM_USE_CACHE`` or ``--no-llm-cache`` says otherwise.

    Notes
    -----
    `repair()` never raises. Every failure path falls back to the mock repairer
    for that cell and is counted in `stats`.

    Keys rotate on HTTP 429 — see the module docstring. `active_api_key` is the
    credential the next attempt will use; it is never logged or serialised.
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        fallback: Optional[MockRepairer] = None,
        top_k: int = 3,
        seed: int = 42,
        cache: Union["DiskCache", str, Path, None] = None,
        client: Optional[LLMTransport] = None,
        use_cache: Optional[bool] = None,
    ) -> None:
        self.config = config if config is not None else LLMConfig.from_env()
        self.top_k = top_k
        self.seed = seed
        self.fallback = fallback if fallback is not None else MockRepairer(seed=seed, top_k=top_k)
        self.use_cache = self.config.use_cache if use_cache is None else bool(use_cache)
        self._stats = LLMStats()
        self._transport: Optional[LLMTransport] = client
        # Rotation state. The index persists across cells, so a key that has
        # already been throttled is not tried again for the next cell.
        self._keys: List[str] = list(self.config.key_pool)
        self._key_index: int = 0

        if cache is None:
            self.cache: Any = DiskCache(self.config.cache_dir)
        elif isinstance(cache, (str, Path)):
            self.cache = DiskCache(cache)
        else:
            self.cache = cache

    # ------------------------------------------------------------------ #
    # Public API (identical to MockRepairer)                              #
    # ------------------------------------------------------------------ #

    @property
    def stats(self) -> LLMStats:
        """Live counters: calls, cache hits, successes, fallbacks by reason."""
        return self._stats

    def repair(self, context: RepairContext) -> RepairResult:
        """
        Generate repair candidates for one suspicious cell.

        Parameters
        ----------
        context : RepairContext
            Built by `build_repair_context()` — contains NO clean values. The
            outgoing request body is derived from this object alone.

        Returns
        -------
        RepairResult
            `repair_method == "llm"` on success, or
            `"llm->mock (<reason>)"` when this cell fell back. Never raises.
        """
        self._stats.cells += 1

        if not self.config.is_configured:
            return self._fallback(context, FallbackReason.NOT_CONFIGURED)

        try:
            result_or_reason = self._attempt_llm(context)
        except Exception as exc:  # noqa: BLE001 - fallback must be total
            logger.warning(
                "Unexpected LLM repair failure for [%s row=%d]: %s",
                context.column, context.row_id,
                self._scrub(exc),
            )
            return self._fallback(context, FallbackReason.REQUEST_FAILED)

        if isinstance(result_or_reason, FallbackReason):
            return self._fallback(context, result_or_reason)

        self._stats.successes += 1
        return result_or_reason

    def repair_batch(
        self,
        records: List[DetectionRecord],
        dirty_df: pd.DataFrame,
        profile: DatasetProfile,
    ) -> List[RepairResult]:
        """
        Repair all detection records in a batch, in input order.

        Mirrors `MockRepairer.repair_batch` exactly: one `RepairResult` per
        input record, same order, built from `build_repair_context()`. No
        `clean_df` parameter exists, by design.
        """
        results: List[RepairResult] = []
        for rec in records:
            ctx = build_repair_context(rec, dirty_df, profile)
            results.append(self.repair(ctx))
        logger.info(
            "LLM batch repair complete for '%s': %d cells, %d via LLM, %d fallbacks.",
            profile.name, len(results), self._stats.successes, self._stats.fallbacks,
        )
        return results

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def build_payload(self, context: RepairContext) -> Dict[str, Any]:
        """
        Build the OpenAI-compatible request body for one cell.

        LEAKAGE: every field here comes from `context` (dirty data + aggregate
        profile stats) or from static prompt text. No clean value, no other
        row, no full dataset.
        """
        system_prompt = SYSTEM_PROMPT.format(top_k=self.top_k)
        user_prompt = build_repair_prompt(context, top_k=self.top_k)
        return {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

    def _attempt_llm(self, context: RepairContext) -> Union[RepairResult, FallbackReason]:
        """Cache → transport → validate. Returns a result or a fallback reason."""
        payload = self.build_payload(context)
        prompt_text = "\n\n".join(m["content"] for m in payload["messages"])
        cache_key = self._cache_key(prompt_text)

        cached = self._cache_get(cache_key)
        if cached is not None:
            self._stats.cache_hits += 1
            response_text: str = cached
            from_cache = True
        else:
            self._stats.llm_calls += 1
            try:
                response_text = self._send_with_retries(payload)
            except LLMTimeoutError as exc:
                logger.warning(
                    "LLM timeout for [%s row=%d]: %s",
                    context.column, context.row_id, self._scrub(exc),
                )
                return FallbackReason.TIMEOUT
            except LLMError as exc:
                logger.warning(
                    "LLM request failed for [%s row=%d]: %s",
                    context.column, context.row_id, self._scrub(exc),
                )
                return FallbackReason.REQUEST_FAILED
            from_cache = False

        candidates, reason = parse_llm_response_detailed(
            response_text, context, top_k=self.top_k
        )
        if reason is not None:
            logger.warning(
                "Rejected LLM response for [%s row=%d]: %s",
                context.column, context.row_id, reason,
            )
            return _PARSE_REASON_MAP.get(reason, FallbackReason.MALFORMED_RESPONSE)

        if not from_cache and self.use_cache:
            self._cache_set(cache_key, response_text)

        return self._make_result(context, candidates, from_cache=from_cache)

    def _send_with_retries(self, payload: Dict[str, Any]) -> str:
        """
        Send one request: rotate keys on 429, then back off, then give up.

        Order of preference on HTTP 429, cheapest first:

        1. **Rotate.** Advance to a key in the pool that has not been
           rate-limited *during this request* and retry at once. No sleep, and
           the retry budget is untouched — a different credential is not the
           same request being re-sent.
        2. **Back off.** Once every key has 429'd here, fall through to the
           existing exponential backoff on the current key.
        3. **Raise.** The caller turns that into a per-cell mock fallback.

        429 is the only rotation trigger: a 5xx is a provider-side fault that
        affects every key, so rotating on it would burn the whole pool on one
        outage. 400/401/403 and every other permanent status propagate on the
        first attempt — retrying them only wastes quota.
        """
        transport = self._transport_or_build()
        attempt = 0
        # Key indices already throttled for THIS request. Reset per request,
        # because a rate limit is a time window, and the backoff below buys
        # exactly that time.
        throttled: set = set()
        while True:
            attempt += 1
            self._apply_active_key(transport)
            self._stats.transport_attempts += 1
            try:
                return transport.chat_completion(payload, timeout_s=self.config.timeout_s)
            except LLMHTTPError as exc:
                if exc.status_code == RATE_LIMIT_STATUS:
                    self._stats.rate_limited += 1
                    if self._rotate_key(throttled):
                        self._stats.key_rotations += 1
                        logger.info(
                            "Rate limited; rotated to key %d of %d and retrying "
                            "immediately (no backoff).",
                            self._key_index + 1, len(self._keys),
                        )
                        attempt -= 1  # rotation is free: not a backoff retry
                        continue
                retryable = is_retryable_status(exc.status_code)
                if not retryable or attempt > self.config.max_retries:
                    raise
                self._stats.retries += 1
                delay = self.config.retry_backoff_s * (2 ** (attempt - 1))
                logger.info(
                    "Retrying after HTTP %d (attempt %d/%d, backoff %.2fs).",
                    exc.status_code, attempt, self.config.max_retries + 1, delay,
                )
                if delay > 0:
                    time.sleep(delay)

    # ── Key rotation ────────────────────────────────────────────────────────

    @property
    def active_api_key(self) -> Optional[str]:
        """The credential the next attempt will use. Never logged."""
        if self._keys:
            return self._keys[self._key_index]
        return self.config.api_key

    def _apply_active_key(self, transport: Any) -> None:
        """
        Tell the transport which key to use, if it can be told.

        Duck-typed on purpose: `set_api_key` is not part of the
        `LLMTransport` protocol, so a transport (or a test fake) that predates
        rotation keeps working untouched.
        """
        setter = getattr(transport, "set_api_key", None)
        if callable(setter):
            try:
                setter(self.active_api_key)
            except Exception:  # noqa: BLE001 - rotation must never be fatal
                logger.debug("Transport rejected a key switch; keeping the current one.")

    def _rotate_key(self, throttled: set) -> bool:
        """
        Advance to the next key not yet throttled in this request.

        Returns True when the active key changed. False means "no spare key" —
        a single-key pool, or every key already 429'd here — which is the
        signal to fall back to backoff.
        """
        if len(self._keys) <= 1:
            return False
        throttled.add(self._key_index)
        n = len(self._keys)
        for step in range(1, n):
            candidate = (self._key_index + step) % n
            if candidate not in throttled:
                self._key_index = candidate
                return True
        return False

    def _scrub(self, text: Any) -> str:
        """`_redact` against every key in the pool, not just the first."""
        out = _redact(text, self.config.api_key)
        for key in self._keys:
            out = _redact(out, key)
        return out

    def _transport_or_build(self) -> LLMTransport:
        """Return the injected transport, or build the default one lazily."""
        if self._transport is None:
            self._transport = _build_default_transport(self.config)
        return self._transport

    # ── Cache helpers (tolerate any cache-like object) ──────────────────────

    def _cache_key(self, prompt_text: str) -> str:
        maker = getattr(self.cache, "make_key", None)
        if callable(maker):
            return maker(self.config.model, prompt_text)
        return DiskCache.make_key(self.config.model, prompt_text)

    def _cache_get(self, key: str) -> Optional[str]:
        if not self.use_cache or self.cache is None:
            return None
        try:
            return self.cache.get(key)
        except Exception:  # noqa: BLE001 - a broken cache must not break a run
            logger.debug("Cache read failed; treating as a miss.")
            return None

    def _cache_set(self, key: str, response_text: str) -> None:
        if not self.use_cache or self.cache is None:
            return
        try:
            self.cache.set(key, response_text, self.config.model)
        except TypeError:
            try:
                self.cache.set(key, response_text)
            except Exception:  # noqa: BLE001
                logger.debug("Cache write failed; continuing.")
        except Exception:  # noqa: BLE001
            logger.debug("Cache write failed; continuing.")

    # ── Result construction ─────────────────────────────────────────────────

    def _make_result(
        self,
        context: RepairContext,
        candidates: List[RepairCandidate],
        from_cache: bool,
    ) -> RepairResult:
        """Wrap validated candidates in a RepairResult with repair_method='llm'."""
        return RepairResult(
            dataset=context.dataset,
            row_id=context.row_id,
            column=context.column,
            dirty_value=context.dirty_value,
            error_type=context.error_type,
            candidates=candidates,
            repair_method="llm",
            context_used={
                "vocabulary_size": len(context.vocabulary),
                "has_stats": context.column_stats is not None,
                "kind": context.kind.value,
                "model": self.config.model,
                "from_cache": from_cache,
            },
        )

    def _fallback(self, context: RepairContext, reason: FallbackReason) -> RepairResult:
        """
        Repair this one cell with `MockRepairer` and label the result.

        The mock's own method name is preserved in `context_used["mock_method"]`
        so a fallback remains fully explainable in the predictions CSV.
        """
        self._stats.record_fallback(reason)
        mock_result = self.fallback.repair(context)
        context_used = dict(mock_result.context_used)
        context_used["mock_method"] = mock_result.repair_method
        context_used["fallback_reason"] = reason.value
        context_used["model"] = self.config.model
        return replace(
            mock_result,
            repair_method=f"llm->mock ({reason.value})",
            context_used=context_used,
        )
