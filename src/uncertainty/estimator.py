"""
src/uncertainty/estimator.py
-----------------------------
Phase 5: Uncertainty estimation for mock/LLM repair candidates.

PIPELINE POSITION
==================
  dirty data → detection → repair candidates → UNCERTAINTY ESTIMATION → (calibration, Phase 6)

This module computes three distinct uncertainty signals per repaired cell:

  1. Raw confidence    — top candidate's assigned probability
  2. Self-consistency  — agreement rate across N independent repair samples
  3. Entropy           — Shannon entropy of the candidate probability distribution

These signals are kept SEPARATE. They measure different things and will be
combined only in Phase 6 (calibration) or later.

CONCEPTUAL DISTINCTIONS
========================

  Raw confidence
  --------------
  "How probable does the repairer say its top candidate is?"
  Derived directly from the probability distribution output of one repair call.
  Subject to the model's own biases — may be over- or under-confident.
  Not yet calibrated.

  Self-consistency
  ----------------
  "How consistently does repeated sampling produce the same answer?"
  We run the repairer N times with different random seeds for the same cell.
  The fraction of runs that agree on the top candidate is the consistency score.
  High consistency means the answer is stable; low means the model is uncertain.
  Independent of the probability values themselves.

  Entropy
  -------
  "How concentrated or spread out is the candidate probability distribution?"
  Shannon entropy H(p) = −Σ p_i * log(p_i)
  Low H → mass concentrated on one candidate → certain.
  High H → mass spread evenly → uncertain.
  Normalized entropy H / log(K) scales to [0, 1] for any number of candidates K.

DATA LEAKAGE PREVENTION
========================
This module does NOT access clean_df.
clean_df is used ONLY by UncertaintyCorrelationAnalysis in evaluation.py.

Usage
======
    from src.repair.mock_repair import MockRepairer
    from src.uncertainty.estimator import UncertaintyEstimator

    estimator = UncertaintyEstimator(base_seed=42, n_samples=5)
    records = estimator.estimate_batch(repair_results, dirty_df, profile)
    for r in records[:3]:
        print(r)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.data.schema import DatasetProfile
from src.detection.detector import DetectionRecord
from src.repair.mock_repair import (
    MockRepairer,
    RepairResult,
    RepairCandidate,
    build_repair_context,
)

logger = logging.getLogger(__name__)

# Default number of samples for self-consistency estimation
_DEFAULT_N_SAMPLES: int = 5

# Default top-k passed to internal samplers during self-consistency
_SELF_CONSISTENCY_TOP_K: int = 3


# ═══════════════════════════════════════════════════════════════════════════════
# Output types
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SelfConsistencyResult:
    """
    Result of N repeated repair samples for a single cell.

    Attributes
    ----------
    n_samples : int
        Number of independent repair attempts.
    most_common_candidate : Any
        The candidate that appeared most often across all samples.
    consistency_score : float
        Frequency of the most common candidate: count / n_samples.
        Range: [1/n_samples, 1.0]. High score → stable answer.
    vote_counts : dict
        Maps str(candidate_value) → number of times it was the top pick.
    candidate_frequencies : dict
        Maps str(candidate_value) → fraction of runs (vote / n_samples).
    """

    n_samples: int
    most_common_candidate: Any
    consistency_score: float
    vote_counts: Dict[str, int] = field(default_factory=dict)
    candidate_frequencies: Dict[str, float] = field(default_factory=dict)


@dataclass
class UncertaintyRecord:
    """
    All uncertainty signals for one repaired cell.

    These signals are raw and uncalibrated. They will be passed to the
    calibration module (Phase 6) for probability calibration.

    Attributes
    ----------
    dataset : str
    row_id : int
    column : str
    dirty_value : Any
    error_type : str
    repair_method : str
    top_candidate : Any
        Value of the highest-probability candidate.
    raw_confidence : float
        Probability assigned to top_candidate by a single repair call.
        Range: (0, 1]. NOT yet calibrated.
    self_consistency : float
        Fraction of N repeated samples that agree on the top candidate.
        Range: [1/n_samples, 1.0].
    entropy : float
        Shannon entropy H(p) = −Σ p_i * log(p_i) over candidates.
        Uses natural logarithm. Range: [0, log(K)].
    normalized_entropy : float
        entropy / log(K), scaled to [0, 1].
        0 = perfectly certain (all mass on one candidate).
        1 = maximally uncertain (uniform distribution).
    n_samples : int
        Number of self-consistency samples used.
    vote_counts : dict
        Maps str(candidate_value) → vote count from self-consistency.
    candidate_frequencies : dict
        Maps str(candidate_value) → frequency from self-consistency.
    n_candidates : int
        Number of candidates in the single-call repair output.
    """

    dataset: str
    row_id: int
    column: str
    dirty_value: Any
    error_type: str
    repair_method: str
    top_candidate: Any
    raw_confidence: float
    self_consistency: float
    entropy: float
    normalized_entropy: float
    n_samples: int
    vote_counts: Dict[str, int] = field(default_factory=dict)
    candidate_frequencies: Dict[str, float] = field(default_factory=dict)
    n_candidates: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "row_id": self.row_id,
            "column": self.column,
            "dirty_value": self.dirty_value,
            "error_type": self.error_type,
            "top_candidate": self.top_candidate,
            "raw_confidence": round(self.raw_confidence, 4),
            "self_consistency": round(self.self_consistency, 4),
            "entropy": round(self.entropy, 4),
            "normalized_entropy": round(self.normalized_entropy, 4),
            "n_samples": self.n_samples,
            "n_candidates": self.n_candidates,
        }

    def __repr__(self) -> str:
        return (
            f"UncertaintyRecord(col={self.column!r}, row={self.row_id}, "
            f"top={self.top_candidate!r}, "
            f"conf={self.raw_confidence:.3f}, "
            f"cons={self.self_consistency:.3f}, "
            f"H_n={self.normalized_entropy:.3f})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# UncertaintyEstimator
# ═══════════════════════════════════════════════════════════════════════════════


class UncertaintyEstimator:
    """
    Computes uncertainty signals for a set of repair results.

    Parameters
    ----------
    base_seed : int
        Base random seed. Self-consistency samples use seeds
        base_seed, base_seed + 1, ..., base_seed + (n_samples - 1).
        Ensures reproducibility across runs.
    n_samples : int
        Number of repair samples for self-consistency estimation.
    top_k : int
        Number of candidates per repair call (used internally).
    """

    def __init__(
        self,
        base_seed: int = 42,
        n_samples: int = _DEFAULT_N_SAMPLES,
        top_k: int = _SELF_CONSISTENCY_TOP_K,
    ) -> None:
        self.base_seed = base_seed
        self.n_samples = n_samples
        self.top_k = top_k

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def estimate(
        self,
        result: RepairResult,
        dirty_df: pd.DataFrame,
        profile: DatasetProfile,
    ) -> UncertaintyRecord:
        """
        Compute all uncertainty signals for a single repair result.

        Parameters
        ----------
        result : RepairResult
            Output of MockRepairer.repair() for one cell. Contains NO clean values.
        dirty_df : pd.DataFrame
            The dirty DataFrame (used to rebuild context for self-consistency sampling).
        profile : DatasetProfile
            Dataset schema (column kinds, numerical stats).

        Returns
        -------
        UncertaintyRecord
        """
        # ── 1. Raw confidence ────────────────────────────────────────────
        top_candidate, raw_confidence = _compute_raw_confidence(result)

        # ── 2. Entropy ───────────────────────────────────────────────────
        entropy, normalized_entropy = _compute_entropy(result.candidates)

        # ── 3. Self-consistency ──────────────────────────────────────────
        sc = self._compute_self_consistency(result, dirty_df, profile)

        logger.debug(
            "Uncertainty [%s/%s row=%d]: conf=%.3f cons=%.3f H_n=%.3f",
            result.column, result.error_type, result.row_id,
            raw_confidence, sc.consistency_score, normalized_entropy,
        )

        return UncertaintyRecord(
            dataset=result.dataset,
            row_id=result.row_id,
            column=result.column,
            dirty_value=result.dirty_value,
            error_type=result.error_type,
            repair_method=result.repair_method,
            top_candidate=top_candidate,
            raw_confidence=raw_confidence,
            self_consistency=sc.consistency_score,
            entropy=entropy,
            normalized_entropy=normalized_entropy,
            n_samples=sc.n_samples,
            vote_counts=sc.vote_counts,
            candidate_frequencies=sc.candidate_frequencies,
            n_candidates=len(result.candidates),
        )

    def estimate_batch(
        self,
        results: List[RepairResult],
        dirty_df: pd.DataFrame,
        profile: DatasetProfile,
    ) -> List[UncertaintyRecord]:
        """
        Compute uncertainty signals for a batch of repair results.

        Parameters
        ----------
        results : list[RepairResult]
        dirty_df : pd.DataFrame
        profile : DatasetProfile

        Returns
        -------
        list[UncertaintyRecord]
        """
        records = []
        for res in results:
            records.append(self.estimate(res, dirty_df, profile))
        logger.info(
            "UncertaintyEstimator: processed %d cells for '%s'.",
            len(records),
            profile.name,
        )
        return records

    # ------------------------------------------------------------------ #
    # Self-consistency (internal)                                         #
    # ------------------------------------------------------------------ #

    def _compute_self_consistency(
        self,
        result: RepairResult,
        dirty_df: pd.DataFrame,
        profile: DatasetProfile,
    ) -> SelfConsistencyResult:
        """
        Run n_samples independent repair attempts for the same cell.

        Each attempt uses a MockRepairer with seed = base_seed + i, so that
        different attempts experience different noise realizations for the
        same cell.

        The cell identity is encoded in the repair result fields, which are
        used to reconstruct a DetectionRecord for context building.

        Returns
        -------
        SelfConsistencyResult
        """
        # Reconstruct a minimal DetectionRecord so we can call build_repair_context
        pseudo_record = DetectionRecord(
            dataset=result.dataset,
            row_id=result.row_id,
            column=result.column,
            dirty_value=result.dirty_value,
            error_type=result.error_type,
            detection_score=1.0,
            detection_method="self_consistency_reconstruction",
            reason="self_consistency_sampling",
        )

        vote_list: List[Any] = []
        for i in range(self.n_samples):
            # Different seed per sample → different noise realization
            repairer = MockRepairer(seed=self.base_seed + i, top_k=self.top_k)
            ctx = build_repair_context(pseudo_record, dirty_df, profile)
            sample_result = repairer.repair(ctx)
            top_val = (
                sample_result.top_candidate.value
                if sample_result.candidates
                else None
            )
            vote_list.append(top_val)

        # Count votes
        vote_counts: Dict[str, int] = {}
        val_by_str: Dict[str, Any] = {}  # str representation → original value
        for val in vote_list:
            key = str(val)
            vote_counts[key] = vote_counts.get(key, 0) + 1
            if key not in val_by_str:
                val_by_str[key] = val

        # Most common candidate
        most_common_str = max(vote_counts, key=lambda k: vote_counts[k])
        most_common_val = val_by_str[most_common_str]
        most_common_count = vote_counts[most_common_str]
        consistency_score = most_common_count / self.n_samples

        candidate_frequencies = {
            k: round(v / self.n_samples, 6) for k, v in vote_counts.items()
        }

        return SelfConsistencyResult(
            n_samples=self.n_samples,
            most_common_candidate=most_common_val,
            consistency_score=round(consistency_score, 6),
            vote_counts=vote_counts,
            candidate_frequencies=candidate_frequencies,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone computation functions (also used in tests)
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_raw_confidence(result: RepairResult) -> Tuple[Any, float]:
    """
    Extract the top candidate and its raw confidence from a RepairResult.

    Raw confidence = the probability assigned to the top candidate by a
    single repair call. This is NOT calibrated — the model may be
    overconfident or underconfident.

    Parameters
    ----------
    result : RepairResult

    Returns
    -------
    (top_candidate_value, raw_confidence)
    """
    if not result.candidates:
        return None, 0.0
    top = result.candidates[0]
    return top.value, top.probability


def _compute_entropy(candidates: List[RepairCandidate]) -> Tuple[float, float]:
    """
    Compute Shannon entropy and normalized entropy over candidate probabilities.

    Shannon entropy
    ---------------
    H(p) = −Σ p_i * ln(p_i)   (natural logarithm)

    Interpretation:
    - H = 0            → all probability mass on one candidate → maximally certain
    - H = ln(K)        → uniform distribution over K candidates → maximally uncertain

    Normalized entropy
    ------------------
    H_normalized = H / ln(K)

    Scales to [0, 1] regardless of the number of candidates K.
    - 0 → certain (all mass on one candidate)
    - 1 → maximally uncertain (uniform distribution)

    Edge cases
    ----------
    - K = 0 → H = 0, H_normalized = 0
    - K = 1 → H = 0, H_normalized = 0  (only one option, so no uncertainty)
    - p_i = 0 → contribution 0 * ln(0) = 0 (by continuity of x*ln(x) at x=0)

    Parameters
    ----------
    candidates : list[RepairCandidate]

    Returns
    -------
    (entropy, normalized_entropy)
    """
    if not candidates:
        return 0.0, 0.0

    k = len(candidates)
    if k == 1:
        return 0.0, 0.0

    # Shannon entropy H = -sum(p * ln(p)) for p > 0
    h = 0.0
    for c in candidates:
        p = c.probability
        if p > 0.0:
            h -= p * math.log(p)  # natural log

    # Maximum entropy for K candidates is ln(K)
    h_max = math.log(k)
    h_normalized = h / h_max if h_max > 0 else 0.0

    return round(h, 6), round(h_normalized, 6)


def compute_entropy_from_probs(probs: List[float]) -> Tuple[float, float]:
    """
    Compute Shannon entropy and normalized entropy from a raw probability list.

    Convenience function accepting a plain list of floats rather than
    RepairCandidate objects. Automatically normalises probabilities if they
    do not sum to 1.

    Parameters
    ----------
    probs : list[float]
        Probability values. Zero values are handled safely (0 * log(0) = 0).

    Returns
    -------
    (entropy, normalized_entropy)
    """
    if not probs:
        return 0.0, 0.0

    # Normalise in case input is not already a valid distribution
    total = sum(p for p in probs if p > 0)
    if total <= 0:
        return 0.0, 0.0

    normalised = [p / total for p in probs if p > 0]
    k = len(normalised)
    if k == 1:
        return 0.0, 0.0

    h = -sum(p * math.log(p) for p in normalised if p > 0)
    h_max = math.log(k)
    h_normalized = h / h_max if h_max > 0 else 0.0

    return round(h, 6), round(h_normalized, 6)
