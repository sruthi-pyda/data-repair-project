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
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

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
