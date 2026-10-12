"""
src/repair/prompts.py
----------------------
LLM prompt templates for the real-LLM repair phase (Phase 12).

PURPOSE
=======
This module defines the prompt structures that will be sent to a real LLM
(e.g., Claude claude-sonnet-4-6) when MOCK_MODE=false.

In Phase 4 (current), MockRepairer uses heuristics instead of sending these
prompts. The templates are authored here so that Phase 12 can drop in the
real LLM without changing the rest of the pipeline.

DATA LEAKAGE PREVENTION
========================
Prompts are built from RepairContext, which contains NO clean values.
The LLM must infer the correct repair from:
  - the dirty value itself
  - the error type
  - the column vocabulary (from dirty data)
  - the row context (other column values in the same row)
  - column-level statistics (aggregate, not per-cell)

Usage
======
    from src.repair.prompts import build_repair_prompt, parse_llm_response

    prompt = build_repair_prompt(context)
    # response = llm.generate(prompt)  # Phase 12
    # candidates = parse_llm_response(response, context)

TWO PARSERS (Phase 12)
=======================
`parse_llm_response` is the original, lenient parser: it drops individual bad
candidates and returns a bare list, so a caller cannot tell *why* a response
produced nothing. Its signature and behaviour are frozen — existing tests pin
them.

`parse_llm_response_detailed` is the strict parser used by
`src.repair.llm_repair.LLMRepairer`. It returns `(candidates, reason)` where
`reason` is `None` on success or one of the `PARSE_*` codes below, which map
1:1 onto `FallbackReason`. It *rejects* the whole response (rather than
silently dropping candidates) when any probability is NaN/non-positive/>1, when
the probabilities do not sum to approximately 1 before truncation, when a value
cannot be cast to the column dtype, or when the JSON shape is wrong. It
*repairs* two harmless deviations instead: exact duplicate values are collapsed
(their probabilities summed) and more than `top_k` candidates are truncated to
the top `top_k`, after which probabilities are renormalised.

    candidates, reason = parse_llm_response_detailed(response, context, top_k=3)
    if reason is not None:
        ...  # fall back to MockRepairer for this cell, counting `reason`
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

from src.data.schema import ColumnKind
from src.repair.mock_repair import RepairCandidate, RepairContext


# ═══════════════════════════════════════════════════════════════════════════════
# System prompt
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a data repair assistant. You will be given a description of a dirty
data cell and contextual information about the column and row. Your task is to
propose up to {top_k} repair candidates for the corrupted value, ordered by
descending probability (most likely first).

Rules:
1. Return ONLY valid JSON matching the schema below. No preamble or explanation.
2. Probabilities must be positive and sum to exactly 1.0.
3. Each candidate value must be a valid data type for the column.
4. Do NOT use the clean reference data — you must infer from context.
5. If the error is a missing value, propose the most likely imputation.
6. If the value appears to be a digit-duplication error (e.g. 33 instead of 3),
   correct it.

Response schema:
{{
  "candidates": [
    {{"value": <corrected_value>, "probability": <float>}},
    ...
  ],
  "reasoning": "<one-sentence explanation>"
}}
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Prompt builders
# ═══════════════════════════════════════════════════════════════════════════════


def build_repair_prompt(
    context: RepairContext,
    top_k: int = 3,
) -> str:
    """
    Build a repair prompt from a RepairContext.

    The prompt is suitable for sending to any chat-completion LLM
    (e.g., Claude claude-sonnet-4-6 via the Anthropic SDK).

    Parameters
    ----------
    context : RepairContext
        Cell context built from dirty data only. No clean values.
    top_k : int
        Maximum candidates the LLM should propose.

    Returns
    -------
    str
        Full prompt text (user turn). The system turn is SYSTEM_PROMPT.
    """
    parts: List[str] = []

    # Header
    parts.append(f"Dataset   : {context.dataset}")
    parts.append(f"Column    : {context.column}  (type: {context.kind.value}, dtype: {context.dtype})")
    parts.append(f"Error type: {context.error_type}")
    parts.append(f"Row ID    : {context.row_id}")
    parts.append(f"Dirty value: {_fmt_value(context.dirty_value)}")
    parts.append("")

    # Column vocabulary
    if context.vocabulary:
        vocab_str = ", ".join(context.vocabulary[:20])
        parts.append(f"Known valid values (top-20 by frequency): [{vocab_str}]")
    else:
        parts.append("Known valid values: [N/A — numerical column]")

    # Numerical stats
    if context.column_stats:
        s = context.column_stats
        parts.append(
            f"Column statistics: min={s['min']}, max={s['max']}, "
            f"median={s['median']}, Q25={s['q25']}, Q75={s['q75']}"
        )

    # Row context
    if context.row_context:
        parts.append("")
        parts.append("Other values in this row (for context):")
        for col, val in list(context.row_context.items())[:8]:
            parts.append(f"  {col}: {_fmt_value(val)}")

    # Error-specific guidance
    parts.append("")
    parts.append(_error_guidance(context))

    # Request
    parts.append("")
    parts.append(
        f"Propose up to {top_k} repair candidates as JSON following the schema above."
    )

    return "\n".join(parts)


def build_batch_prompt(
    contexts: List[RepairContext],
    top_k: int = 3,
) -> str:
    """
    Build a single prompt for repairing multiple cells at once (batch mode).

    Some LLMs support multi-cell repair in a single call, reducing API costs.
    Each cell is separated by a numbered section.

    Parameters
    ----------
    contexts : list[RepairContext]
    top_k : int

    Returns
    -------
    str
    """
    parts: List[str] = [
        f"Repair the following {len(contexts)} corrupted cells.",
        "For each cell, return a JSON array element with 'cell_id', 'candidates', and 'reasoning'.",
        "",
    ]
    for i, ctx in enumerate(contexts):
        parts.append(f"--- Cell {i + 1} ---")
        parts.append(build_repair_prompt(ctx, top_k=top_k))
        parts.append("")
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Response parser
# ═══════════════════════════════════════════════════════════════════════════════


def parse_llm_response(
    response_text: str,
    context: RepairContext,
    top_k: int = 3,
) -> List[RepairCandidate]:
    """
    Parse a JSON LLM response into a list of RepairCandidates.

    Validates that:
    - probabilities are positive floats
    - probabilities sum to approximately 1.0
    - at most top_k candidates are returned

    Falls back to an empty list if parsing fails.

    Parameters
    ----------
    response_text : str
        Raw string response from the LLM.
    context : RepairContext
        Used for fallback validation (dtype, kind).
    top_k : int

    Returns
    -------
    list[RepairCandidate]
    """
    try:
        data = json.loads(_extract_json_block(response_text))
        raw_candidates = data.get("candidates", [])
    except (json.JSONDecodeError, AttributeError, KeyError):
        return []

    candidates: List[RepairCandidate] = []
    for item in raw_candidates[:top_k]:
        val = item.get("value")
        prob = item.get("probability")
        if not isinstance(prob, (int, float)) or prob <= 0:
            continue
        # Cast value to expected Python type
        val = _cast_value(val, context)
        candidates.append(RepairCandidate(value=val, probability=float(prob)))

    if not candidates:
        return []

    # Normalise to sum to 1.0 in case LLM probabilities are slightly off
    total = sum(c.probability for c in candidates)
    if total <= 0:
        return []

    return [
        RepairCandidate(value=c.value, probability=round(c.probability / total, 6))
        for c in candidates
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Strict response parser (Phase 12) — reports WHY a response was rejected
# ═══════════════════════════════════════════════════════════════════════════════

# Reason codes. Plain strings rather than an enum so this module stays free of
# any dependency on src.repair.llm_repair (which imports this one). The values
# are identical to the corresponding FallbackReason values.
PARSE_MALFORMED_RESPONSE: str = "malformed_response"
PARSE_INVALID_PROBABILITIES: str = "invalid_probabilities"
PARSE_EMPTY_CANDIDATES: str = "empty_candidates"

# How far the raw probabilities may stray from 1.0 before the response is
# rejected outright. 0.25 accepts the common sloppiness of a model that returns
# e.g. 0.9 or 1.1 (renormalised below) while still rejecting a response that
# plainly ignored the schema (all 1.0, or percentages like 70/20/10).
PROBABILITY_SUM_TOLERANCE: float = 0.25


def parse_llm_response_detailed(
    response_text: str,
    context: RepairContext,
    top_k: int = 3,
    sum_tolerance: float = PROBABILITY_SUM_TOLERANCE,
) -> Tuple[List[RepairCandidate], Optional[str]]:
    """
    Parse an LLM response strictly, reporting the reason for any rejection.

    Rejects (returns `([], reason)`):
      - `response_text` empty, or not parseable as JSON        → malformed
      - top-level JSON not an object, or no "candidates" key   → malformed
      - "candidates" not a list                                → malformed
      - a candidate that is not an object, or lacks
        "value" / "probability"                                → malformed
      - a value that cannot be cast to the column's dtype
        (including `None`)                                     → malformed
      - "candidates": []                                       → empty
      - a probability that is non-numeric, NaN, infinite,
        <= 0, or > 1                                           → invalid probs
      - probabilities that do not sum to 1 within
        `sum_tolerance` (checked before truncation)            → invalid probs

    Repairs (returns `(candidates, None)`):
      - exact duplicate values are collapsed, probabilities summed
      - more than `top_k` candidates are truncated to the top `top_k`
      - the kept probabilities are renormalised to sum to 1.0

    Candidates are returned in descending probability order, so
    `candidates[0]` is always the top candidate.

    LEAKAGE: `context` is used only for dtype casting. No clean value is
    consulted here or anywhere else in this module.

    Parameters
    ----------
    response_text : str
        Raw assistant message text.
    context : RepairContext
        Supplies the expected dtype for casting.
    top_k : int
        Maximum candidates to keep.
    sum_tolerance : float
        Allowed absolute deviation of the raw probability sum from 1.0.

    Returns
    -------
    (list[RepairCandidate], str | None)
        Validated candidates and `None`, or `[]` and a `PARSE_*` reason code.
    """
    if not isinstance(response_text, str) or not response_text.strip():
        return [], PARSE_MALFORMED_RESPONSE

    try:
        data = json.loads(_extract_json_block(response_text))
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
        return [], PARSE_MALFORMED_RESPONSE

    if not isinstance(data, dict) or "candidates" not in data:
        return [], PARSE_MALFORMED_RESPONSE

    raw_candidates = data["candidates"]
    if not isinstance(raw_candidates, list):
        return [], PARSE_MALFORMED_RESPONSE
    if not raw_candidates:
        return [], PARSE_EMPTY_CANDIDATES

    parsed: List[Tuple[Any, float]] = []
    for item in raw_candidates:
        if not isinstance(item, dict):
            return [], PARSE_MALFORMED_RESPONSE
        if "value" not in item or "probability" not in item:
            return [], PARSE_MALFORMED_RESPONSE

        prob = item["probability"]
        if isinstance(prob, bool) or not isinstance(prob, (int, float)):
            return [], PARSE_INVALID_PROBABILITIES
        prob = float(prob)
        if math.isnan(prob) or math.isinf(prob) or prob <= 0.0 or prob > 1.0:
            return [], PARSE_INVALID_PROBABILITIES

        ok, value = _cast_value_strict(item["value"], context)
        if not ok:
            return [], PARSE_MALFORMED_RESPONSE

        parsed.append((value, prob))

    if not parsed:
        return [], PARSE_EMPTY_CANDIDATES

    merged = _collapse_duplicates(parsed)

    total = sum(p for _, p in merged)
    if total <= 0 or math.isnan(total) or math.isinf(total):
        return [], PARSE_INVALID_PROBABILITIES
    if abs(total - 1.0) > sum_tolerance:
        return [], PARSE_INVALID_PROBABILITIES

    kept = sorted(merged, key=lambda pair: pair[1], reverse=True)[: max(1, int(top_k))]
    kept_total = sum(p for _, p in kept)
    if kept_total <= 0:
        return [], PARSE_INVALID_PROBABILITIES

    candidates = [
        RepairCandidate(value=value, probability=round(prob / kept_total, 6))
        for value, prob in kept
    ]
    if not candidates:
        return [], PARSE_EMPTY_CANDIDATES
    return candidates, None


def _collapse_duplicates(
    parsed: List[Tuple[Any, float]],
) -> List[Tuple[Any, float]]:
    """
    Merge exact duplicate values, summing their probabilities.

    Duplicates are judged *after* dtype casting, so a model returning 3 and
    "3" for an int column produces one candidate, not two. The first
    occurrence's value object is kept, preserving the model's ordering for
    equal probabilities.
    """
    order: List[Any] = []
    totals: Dict[Any, float] = {}
    values: Dict[Any, Any] = {}

    for value, prob in parsed:
        try:
            key: Any = ("h", value)
            hash(key)
        except TypeError:
            key = ("s", str(value))
        if key not in totals:
            order.append(key)
            totals[key] = 0.0
            values[key] = value
        totals[key] += prob

    return [(values[key], totals[key]) for key in order]


def _cast_value_strict(val: Any, context: RepairContext) -> Tuple[bool, Any]:
    """
    Cast an LLM-returned value to the column's dtype, reporting failure.

    Unlike `_cast_value`, which silently returns the original value when
    casting fails, this returns `(False, None)` so the caller can reject the
    response and fall back. `None` is never a usable repair value, and
    containers (lists/dicts) are never valid cell values.

    Returns
    -------
    (bool, Any)
        `(True, casted)` or `(False, None)`.
    """
    if val is None or isinstance(val, (list, dict, tuple, set)):
        return False, None

    dtype = (context.dtype or "").lower()
    try:
        if "int" in dtype:
            as_float = float(val)
            if math.isnan(as_float) or math.isinf(as_float):
                return False, None
            return True, int(round(as_float))
        if "float" in dtype:
            as_float = float(val)
            if math.isnan(as_float) or math.isinf(as_float):
                return False, None
            return True, as_float
        if "bool" in dtype:
            if isinstance(val, bool):
                return True, val
            if isinstance(val, (int, float)) and val in (0, 1):
                return True, bool(val)
            text = str(val).strip().lower()
            if text in ("true", "false"):
                return True, text == "true"
            return False, None
    except (TypeError, ValueError):
        return False, None

    # Text-like columns ('object', 'str', 'category'): accept any scalar.
    return True, val


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _fmt_value(val: Any) -> str:
    """Format a value for display in a prompt."""
    if val is None:
        return "NULL / missing"
    return repr(val)


def _error_guidance(context: RepairContext) -> str:
    """Return error-type-specific guidance for the LLM."""
    et = context.error_type
    kind = context.kind

    if et == "missing":
        if kind in (ColumnKind.CATEGORICAL, ColumnKind.BINARY):
            return (
                "The cell is missing. Propose the most likely categorical value "
                "based on the column's value distribution and row context."
            )
        else:
            return (
                "The cell is missing. Propose a plausible numerical imputation "
                "using the column statistics and row context."
            )

    elif et == "novel_category":
        return (
            "The value is not in the known vocabulary. It may be a typo, "
            "abbreviation, or reformatting of a valid category. "
            "Propose the closest matching known value."
        )

    elif et == "out_of_range":
        return (
            "The value is outside the valid range for this column. "
            "Common causes: digit duplication (e.g. 33 instead of 3, 88 instead of 8). "
            "Try dividing by 11 or 10 first. Fall back to the column median."
        )

    elif et == "format_error":
        if kind in (ColumnKind.CATEGORICAL, ColumnKind.MIXED):
            return (
                "The value has an unexpected format for a categorical column. "
                "Identify the closest known vocabulary item."
            )
        else:
            return (
                "The value contains non-numeric characters in a numerical column. "
                "Extract the numeric portion and validate against column range."
            )

    elif et == "statistical_outlier":
        return (
            "The value is a statistical outlier. It may be a data entry error. "
            "Propose the most likely corrected value using column statistics."
        )

    return "Propose the most likely corrected value for this corrupted cell."


def _extract_json_block(text: str) -> str:
    """Extract JSON from a response that may contain surrounding text."""
    text = text.strip()
    # Look for JSON block delimiters
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        return text[start:end].strip()
    if "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        return text[start:end].strip()
    # Try to find a raw JSON object
    brace_start = text.find("{")
    if brace_start != -1:
        return text[brace_start:]
    return text


def _cast_value(val: Any, context: RepairContext) -> Any:
    """
    Attempt to cast the LLM-returned value to the expected Python type.
    Falls back to the original if casting fails.
    """
    if val is None:
        return val
    dtype = context.dtype
    try:
        if "int" in dtype:
            return int(round(float(val)))
        elif "float" in dtype:
            return float(val)
    except (TypeError, ValueError):
        pass
    return val
